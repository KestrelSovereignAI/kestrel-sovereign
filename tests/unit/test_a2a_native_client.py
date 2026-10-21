"""Real SDK bindings, trusted cards, and durable outbound delivery evidence."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from a2a.client.errors import A2AClientError
from a2a.types import (
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from a2a.utils.signing import InvalidSignaturesError, create_agent_card_signer
from google.protobuf.json_format import MessageToDict, ParseDict
from starlette.applications import Starlette

from kestrel_sovereign.a2a.native_card import (
    REQUIRED_NATIVE_EXTENSIONS,
    create_native_agent_card,
)
from kestrel_sovereign.a2a.native_client import (
    NativePeerProtocolError,
    NativePeerSession,
    NativeSenderInterceptor,
)
from kestrel_sovereign.a2a.native_routes import NativeA2AHTTP
from kestrel_sovereign.a2a.native_sender import native_request_digest
from kestrel_sovereign.a2a.request_handler import NativeA2ARequestHandler
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatchConflict,
    OutboundStore,
)
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.identity.runtime_identity import AgentIdentity
from tests.utils.native_a2a import signing_identity
from tests.utils.native_a2a_security import NOW, PRINCIPAL, SENDER


@pytest.fixture(scope="module")
def native_signing():
    pair, document = signing_identity(SENDER)
    document["verificationMethod"][0]["id"] = SENDER + "#rotated-classical"
    document["verificationMethod"][1]["id"] = SENDER + "#rotated-pq"
    document["verificationMethod"].reverse()
    return pair, document


@pytest.fixture(scope="module")
def recipient_identity():
    pair, document = signing_identity("did:web:peer.test:recipient")
    return AgentIdentity(
        hybrid_keypair=pair,
        new_did=document["id"],
        new_verification_methods=document["verificationMethod"],
    ), document


@pytest_asyncio.fixture(params=["JSONRPC", "HTTP+JSON"])
async def peer_session(native_request_admission, recipient_identity, request):
    a = native_request_admission
    identity, document = recipient_identity
    card = create_native_agent_card(
        identity=identity,
        scope=a.scope,
        routing_key=a.guard.routing_key,
        public_base_url="https://peer.test",
        name="Peer",
        description="Native peer",
        version="1",
        skills=[
            AgentSkill(
                id="message", name="Message", description="Receive work", tags=["chat"]
            )
        ],
    )
    state = SimpleNamespace(
        drop_ack=False,
        authorized=True,
        checks=0,
        revoke_on_response=False,
        response_error=None,
        taskless=False,
        early_task_response=False,
        received_messages=0,
        received_cancellations=0,
        revoke_on_request=False,
    )

    class PeerHandler(NativeA2ARequestHandler):
        async def on_cancel_task(self, params, context):
            state.received_cancellations += 1
            return await super().on_cancel_task(params, context)

        async def on_message_send(self, params, context):
            state.received_messages += 1
            if state.early_task_response:
                async with self.guard.admit(params, "SendMessage", context) as admitted:
                    return await self._submit(admitted)
            if not state.taskless:
                return await super().on_message_send(params, context)
            async with self.guard.admit(params, "SendMessage", context):
                from kestrel_sovereign.a2a.extensions import correlated_answer_metadata

                return Message(
                    metadata=correlated_answer_metadata(
                        MessageToDict(params.metadata), {}
                    ),
                    message_id=uuid4().hex,
                    role=Role.ROLE_AGENT,
                    context_id=params.message.context_id or uuid4().hex,
                    parts=[Part(text="Standalone native response")],
                )

    binding = NativeA2AHTTP(
        PeerHandler(a.manager, a.guard, card), public_base_url="https://peer.test"
    )
    trace = []

    async def capture(outgoing):
        if state.revoke_on_request:
            state.authorized = False
        trace.append(
            (
                outgoing.method,
                str(outgoing.url),
                json.loads(outgoing.content) if outgoing.content else None,
                outgoing.headers.get("A2A-Version"),
            )
        )

    async def after_response(response):
        if state.response_error is not None:
            error, state.response_error = state.response_error, None
            await response.aclose()
            raise error
        if state.revoke_on_response:
            state.authorized = False
        if state.drop_ack:
            state.drop_ack = False
            await response.aclose()
            raise httpx.ReadError(
                "Acknowledgement lost after recipient committed",
                request=response.request,
            )

    async def authorize():
        state.checks += 1
        if not state.authorized:
            raise PermissionError("Peer scope was revoked")

    sender_identity = AgentIdentity(
        hybrid_keypair=a.signing[0],
        new_did=SENDER,
        new_verification_methods=a.signing[1]["verificationMethod"],
    )
    signer = NativeSenderInterceptor.for_identity(
        sender_identity,
        recipient=a.scope,
        extensions=REQUIRED_NATIVE_EXTENSIONS,
        policy=a.guard.proof_policy,
        clock=lambda: NOW,
        nonce=lambda: uuid4().hex,
    )
    store = OutboundStore(
        a.backend, sender_scope=NativeRecipientScope(PRINCIPAL, "sender-tenant")
    )
    await store.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(Starlette(routes=[binding.route])),
        event_hooks={"request": [capture], "response": [after_response]},
    ) as http:
        interface = next(
            item
            for item in card.supported_interfaces
            if item.protocol_binding == request.param
        )
        arguments = {
            "card": card,
            "signing_did": identity.signing_did,
            "trusted_document": document,
            "recipient": a.scope,
            "interface": interface,
            "signer": signer,
            "http_client": http,
            "outbound_store": store,
            "authorize": authorize,
        }
        session = NativePeerSession(**arguments)
        try:
            yield SimpleNamespace(
                session=session,
                store=store,
                trace=trace,
                state=state,
                admission=a,
                arguments=arguments,
                identity=identity,
                http=http,
            )
        finally:
            await session.close()


@pytest_asyncio.fixture
async def network_peer(peer_session):
    """Real loopback HTTP; HTTPX ASGITransport buffers a streaming response."""
    from a2a.types import AgentCard

    from tests.utils.native_a2a import serve_native_app

    p = peer_session
    async with serve_native_app(p.http._transport.app) as base:
        card = AgentCard.FromString(p.arguments["card"].SerializeToString())
        for interface in card.supported_interfaces:
            interface.url = interface.url.replace("https://peer.test", base, 1)
        card = create_agent_card_signer(
            p.identity.hybrid_keypair.classical.private_key,
            {"kid": p.identity.signing_did + "#key-1", "alg": "EdDSA"},
        )(card)
        selected = next(
            item
            for item in card.supported_interfaces
            if item.protocol_binding == p.arguments["interface"].protocol_binding
        )
        async with httpx.AsyncClient(
            trust_env=False, event_hooks=p.http.event_hooks
        ) as http:
            arguments = p.arguments | {
                "card": card,
                "interface": selected,
                "http_client": http,
            }
            session = NativePeerSession(**arguments)
            try:
                yield SimpleNamespace(
                    **(
                        vars(p)
                        | {"session": session, "http": http, "arguments": arguments}
                    )
                )
            finally:
                await session.close()


def request_for(p):
    return SendMessageRequest(
        tenant=p.admission.scope.tenant,
        message=Message(
            message_id=uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(text="Native work")],
        ),
        configuration={"return_immediately": True},
    )


async def reserve(p):
    return await p.session.reserve(
        request_for(p), dispatch_tool="send_a2a_task", verb="task"
    )


async def test_official_client_records_exact_wire_and_keeps_distinct_ids(peer_session):
    p = peer_session
    original = request_for(p)
    frozen = original.SerializeToString()
    dispatch, wire = await p.session.reserve(
        original, dispatch_tool="send_a2a_task", verb="task"
    )
    receipt = await p.session.send(dispatch, wire)
    assert original.SerializeToString() == frozen
    assert receipt.dispatch.dispatch_id != receipt.response.id
    assert receipt.dispatch.remote_task_id == receipt.response.id
    assert receipt.dispatch.remote_context_id == receipt.response.context_id
    assert receipt.dispatch.state is DispatchState.ACKNOWLEDGED
    assert receipt.dispatch.sender_tenant == "sender-tenant"
    assert p.trace[0][3] == "1.0"
    body = p.trace[0][2]
    if "params" in body:
        body = body["params"]
    # HTTP+JSON carries tenant in its official URL rather than its body.
    body["tenant"] = p.admission.scope.tenant
    actual = ParseDict(body, SendMessageRequest())
    assert native_request_digest(actual, "SendMessage") == dispatch.request_digest
    observed = await p.session.get_task(receipt.dispatch)
    assert observed.response == receipt.response
    assert await p.store.get(dispatch.dispatch_id) == observed.dispatch
    cancelled = await p.session.cancel_task(observed.dispatch)
    assert cancelled.response.id == receipt.response.id
    assert cancelled.response.status.state == TaskState.TASK_STATE_CANCELED
    assert cancelled.dispatch.state is DispatchState.COMPLETE
    assert cancelled.dispatch.cancel_state is DispatchState.COMPLETE
    assert await p.store.get(dispatch.dispatch_id) == cancelled.dispatch
    assert len(p.trace) == 3
    await p.session.close()
    assert not p.http.is_closed, "SDK must borrow the host-owned HTTP client"


async def test_lost_acknowledgement_reconciles_by_read_without_resending(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    p.state.drop_ack = True
    with pytest.raises(A2AClientError, match="Acknowledgement lost"):
        await p.session.send(dispatch, wire)
    unknown = await p.store.get(dispatch.dispatch_id)
    assert unknown.state is DispatchState.UNKNOWN and unknown.remote_task_id is None
    with pytest.raises(OutboundDispatchConflict, match="cannot be submitted again"):
        await p.session.send(dispatch, wire)
    assert len(p.trace) == 1
    reconciled = await p.session.reconcile(unknown)
    assert reconciled.remote_task_id and reconciled.dispatch_id == dispatch.dispatch_id
    assert reconciled.state is DispatchState.ACKNOWLEDGED
    assert len(p.trace) == 2
    assert p.trace[1][0] == "GET" or p.trace[1][2]["method"] == "ListTasks"


async def test_concurrent_senders_have_only_one_native_submission(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    outcomes = await asyncio.gather(
        p.session.send(dispatch, wire),
        p.session.send(dispatch, wire),
        return_exceptions=True,
    )
    assert sum(isinstance(value, OutboundDispatchConflict) for value in outcomes) == 1
    assert len(p.trace) == 1
    assert (await p.store.get(dispatch.dispatch_id)).remote_task_id


async def test_revocation_after_recipient_commit_retains_uncertainty(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    p.state.revoke_on_response = True
    with pytest.raises(PermissionError, match="revoked"):
        await p.session.send(dispatch, wire)
    unknown = await p.store.get(dispatch.dispatch_id)
    assert unknown.state is DispatchState.UNKNOWN
    with pytest.raises(PermissionError, match="revoked"):
        await p.session.reconcile(unknown)
    assert len(p.trace) == 1


@pytest.mark.parametrize(
    "field",
    ["recipient_id", "tenant", "signing_did", "interface_url", "protocol_binding"],
)
async def test_retained_route_tampering_never_reaches_peer(peer_session, field):
    p = peer_session
    dispatch, wire = await reserve(p)
    with pytest.raises(OutboundDispatchConflict, match="another peer binding"):
        await p.session.send(replace(dispatch, **{field: "other"}), wire)
    assert not p.trace


async def test_changed_message_cannot_use_a_reservation(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    wire.message.parts[0].text = "different"
    with pytest.raises(OutboundDispatchConflict, match="differs from its reservation"):
        await p.session.send(dispatch, wire)
    assert not p.trace
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.RESERVED


async def test_card_signature_is_verified_before_client_creation(peer_session):
    p = peer_session
    p.arguments["card"].name = "attacker"
    with pytest.raises(InvalidSignaturesError):
        NativePeerSession(**p.arguments)
    assert not p.trace


@pytest.mark.parametrize(
    "fault",
    ["draft", "ambiguous", "other-tenant", "required-extension", "boolean-version"],
)
async def test_signed_but_incompatible_card_is_refused(peer_session, fault):
    p = peer_session
    card = p.arguments["card"]
    selected = p.arguments["interface"]
    if fault == "draft":
        selected.protocol_version = "0.3"
    elif fault == "other-tenant":
        selected.tenant = "other"
    elif fault == "ambiguous":
        card.supported_interfaces.append(
            AgentInterface.FromString(selected.SerializeToString())
        )
    elif fault == "boolean-version":
        card.capabilities.extensions[0].params["version"] = True
    else:
        card.capabilities.extensions.add(
            uri="https://unknown.test/extension", required=True
        )
    p.arguments["card"] = create_agent_card_signer(
        p.identity.hybrid_keypair.classical.private_key,
        {"kid": p.identity.signing_did + "#key-1", "alg": "EdDSA"},
    )(card)
    with pytest.raises(NativePeerProtocolError):
        NativePeerSession(**p.arguments)
    assert not p.trace


@pytest.mark.parametrize(
    "fault", ["redirects", "authorization", "cookie", "api-key", "auth"]
)
async def test_peer_client_refuses_ambient_operator_credentials(peer_session, fault):
    p = peer_session
    if fault == "redirects":
        p.http.follow_redirects = True
    elif fault == "auth":
        p.http.auth = httpx.BasicAuth("operator", "secret")
    elif fault == "cookie":
        p.http.cookies.set("session", "operator")
    else:
        p.http.headers["Authorization" if fault == "authorization" else "X-API-Key"] = (
            "operator"
        )
    with pytest.raises(NativePeerProtocolError, match="ambient credentials"):
        NativePeerSession(**p.arguments)
    assert not p.trace


@pytest.mark.parametrize("kind", ["withdrawal", "cancellation"])
async def test_original_execution_control_survives_sdk_normalization(
    peer_session, kind
):
    p = peer_session
    dispatch, wire = await reserve(p)
    error = (
        ExecutionAuthorityError("original withdrawal")
        if kind == "withdrawal"
        else asyncio.CancelledError("original cancellation")
    )
    p.state.response_error = error
    with pytest.raises(type(error)) as caught:
        await p.session.send(dispatch, wire)
    assert caught.value is error
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.UNKNOWN
    assert len(p.trace) == 1


@pytest.mark.parametrize("operation", ["get_task", "cancel_task"])
async def test_remote_id_cannot_be_fabricated_in_a_dispatch_snapshot(
    peer_session, operation
):
    p = peer_session
    dispatch, wire = await reserve(p)
    receipt = await p.session.send(dispatch, wire)
    forged = replace(receipt.dispatch, remote_task_id="another-task")
    with pytest.raises(OutboundDispatchConflict, match="current committed ledger row"):
        await getattr(p.session, operation)(forged)
    assert len(p.trace) == 1


async def test_credentials_added_after_session_creation_are_refused(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    p.http.headers["Authorization"] = "operator"
    with pytest.raises(NativePeerProtocolError, match="ambient credentials"):
        await p.session.send(dispatch, wire)
    assert not p.trace
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.RESERVED


async def test_restarting_after_submission_intent_uses_reads_only(peer_session):
    p = peer_session
    dispatch, _ = await reserve(p)
    submitting = await p.store.begin_submission(dispatch, operation_id="process-crash")
    assert await p.session.reconcile(submitting) is None
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.UNKNOWN
    assert len(p.trace) == 1
    assert p.trace[0][0] == "GET" or p.trace[0][2]["method"] == "ListTasks"


async def test_absent_configuration_is_frozen_before_sdk_touches_it(peer_session):
    p = peer_session
    original = request_for(p)
    original.ClearField("configuration")
    dispatch, wire = await p.session.reserve(
        original, dispatch_tool="send_a2a_task", verb="task"
    )
    assert not original.HasField("configuration")
    assert wire.HasField("configuration")
    assert native_request_digest(wire, "SendMessage") == dispatch.request_digest
    assert not p.trace


async def test_taskless_native_message_retains_actual_kind_without_fake_task(
    peer_session,
):
    p = peer_session
    p.state.taskless = True
    dispatch, wire = await reserve(p)
    receipt = await p.session.send(dispatch, wire)
    assert isinstance(receipt.response, Message)
    assert receipt.dispatch.response_kind == "message"
    assert receipt.dispatch.state is DispatchState.COMPLETE
    assert receipt.dispatch.remote_task_id is None
    assert not receipt.response.task_id
    assert await p.store.get(dispatch.dispatch_id) == receipt.dispatch


async def test_committed_dispatch_for_another_peer_cannot_use_this_session(
    peer_session,
):
    p = peer_session
    wire = request_for(p)
    interface = AgentInterface.FromString(p.arguments["interface"].SerializeToString())
    interface.url = "https://another-peer.test/a2a"
    dispatch = await p.store.reserve(
        "other-peer-dispatch",
        wire,
        recipient=NativeRecipientScope("did:another-peer", wire.tenant),
        signing_did="did:web:another-peer.test",
        interface=interface,
        dispatch_tool="send_a2a_task",
        verb="task",
    )
    with pytest.raises(OutboundDispatchConflict, match="another peer binding"):
        await p.session.send(dispatch, wire)
    assert not p.trace
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.RESERVED


async def test_request_observer_cannot_spend_revoked_peer_authority(peer_session):
    from a2a.types import ListTasksRequest

    p = peer_session
    dispatch, wire = await reserve(p)
    p.state.revoke_on_request = True
    with pytest.raises(PermissionError, match="revoked"):
        await p.session.send(dispatch, wire)
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.NOT_SENT
    listed = await p.admission.manager.task_store.list_for_creator(
        ListTasksRequest(tenant=p.admission.scope.tenant),
        creator_agent_id=PRINCIPAL,
        recipient_agent_id=p.admission.scope.recipient_id,
    )
    assert not listed.tasks, "Revocation must precede physical recipient admission"


async def test_local_signing_refusal_is_recorded_as_not_sent(peer_session):
    from kestrel_sovereign.a2a.native_sender import NativeSenderProofError

    p = peer_session
    broken = NativePeerSession(
        **(p.arguments | {"signer": replace(p.arguments["signer"], nonce=lambda: "")})
    )
    try:
        dispatch, wire = await broken.reserve(
            request_for(p), dispatch_tool="send_a2a_task", verb="task"
        )
        with pytest.raises(NativeSenderProofError):
            await broken.send(dispatch, wire)
        current = await p.store.get(dispatch.dispatch_id)
        assert current.state is DispatchState.NOT_SENT
        assert not p.trace
        with pytest.raises(OutboundDispatchConflict):
            await p.session.send(current, wire)
        assert not p.trace
    finally:
        await broken.close()


async def test_sdk_retry_cannot_replay_a_send_after_an_uncertain_ack(
    peer_session, monkeypatch
):
    p = peer_session
    dispatch, wire = await reserve(p)
    transport = p.session._client._transport
    original = transport.send_message

    async def sdk_retry(request, *, context=None):
        try:
            return await original(request, context=context)
        except A2AClientError:
            return await original(request, context=context)

    monkeypatch.setattr(transport, "send_message", sdk_retry)
    p.state.drop_ack = True
    with pytest.raises(OutboundDispatchConflict, match="cannot replay"):
        await p.session.send(dispatch, wire)
    assert p.state.received_messages == 1
    assert (await p.store.get(dispatch.dispatch_id)).state is DispatchState.UNKNOWN


async def test_lost_cancellation_ack_preserves_remote_identity_until_read_reconciliation(
    peer_session,
):
    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    p.state.drop_ack = True
    with pytest.raises(A2AClientError, match="Acknowledgement lost"):
        await p.session.cancel_task(accepted.dispatch)
    unknown = await p.store.get(dispatch.dispatch_id)
    assert unknown.state is DispatchState.ACKNOWLEDGED
    assert unknown.cancel_state is DispatchState.UNKNOWN
    assert unknown.remote_task_id == accepted.response.id
    assert unknown.cancel_request_digest and unknown.cancel_operation_id
    count = len(p.trace)
    with pytest.raises(OutboundDispatchConflict, match="already in flight"):
        await p.session.cancel_task(unknown)
    assert len(p.trace) == count
    observed = await p.session.get_task(unknown)
    assert observed.dispatch.state is DispatchState.COMPLETE
    assert observed.dispatch.cancel_state is DispatchState.COMPLETE
    assert observed.dispatch.remote_task_state == "canceled"
    again = await p.session.get_task(observed.dispatch)
    assert again.dispatch.remote_task_state == "canceled"
    assert len(p.trace) == count + 2


async def test_cancellation_revoked_at_native_boundary_is_proven_not_sent(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    p.state.revoke_on_request = True
    with pytest.raises(PermissionError, match="revoked"):
        await p.session.cancel_task(accepted.dispatch)
    retained = await p.store.get(dispatch.dispatch_id)
    assert retained.remote_task_id == accepted.response.id
    assert retained.state is DispatchState.ACKNOWLEDGED
    assert retained.cancel_state is DispatchState.NOT_SENT
    assert len(p.trace) == 2
    # A new explicitly authorized attempt is safe only because the previous
    # native dispatcher was never admitted. No implicit SDK retry is allowed.
    p.state.authorized = True
    p.state.revoke_on_request = False
    canceled = await p.session.cancel_task(retained)
    assert canceled.dispatch.remote_task_state == "canceled"


async def test_stale_read_cannot_overwrite_newer_dispatch_evidence(peer_session):
    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    await p.session.get_task(accepted.dispatch)
    count = len(p.trace)
    with pytest.raises(OutboundDispatchConflict, match="current committed"):
        await p.session.get_task(accepted.dispatch)
    assert len(p.trace) == count


async def test_sdk_cancellation_retry_is_blocked_at_actual_native_dispatch(
    peer_session, monkeypatch
):
    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    transport = p.session._client._transport
    original = transport.cancel_task

    async def sdk_retry(request, *, context=None):
        try:
            return await original(request, context=context)
        except A2AClientError:
            return await original(request, context=context)

    monkeypatch.setattr(transport, "cancel_task", sdk_retry)
    p.state.drop_ack = True
    with pytest.raises(OutboundDispatchConflict, match="cannot replay"):
        await p.session.cancel_task(accepted.dispatch)
    assert p.state.received_cancellations == 1
    retained = await p.store.get(dispatch.dispatch_id)
    assert retained.cancel_state is DispatchState.UNKNOWN
    assert retained.remote_task_id == accepted.response.id


@pytest.mark.parametrize("mime", ["image/png", "bad"])
async def test_client_mime_preflight_refuses_before_reservation_or_http(
    peer_session, mime
):
    from a2a.utils.errors import ContentTypeNotSupportedError, InvalidParamsError

    p = peer_session
    request = request_for(p)
    request.message.parts.append(Part(raw=b"unsupported", media_type=mime))
    with pytest.raises(
        ContentTypeNotSupportedError if mime == "image/png" else InvalidParamsError
    ):
        await p.session.reserve(request, dispatch_tool="send_a2a_task", verb="task")
    assert not p.trace and not await p.store.list_recent()


@pytest.mark.parametrize("kind", ["withdrawal", "cancellation"])
async def test_cancellation_retains_original_execution_control_and_remote_evidence(
    peer_session, kind
):
    p = peer_session
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    error = (
        ExecutionAuthorityError("original cancellation withdrawal")
        if kind == "withdrawal"
        else asyncio.CancelledError("original cancellation control")
    )
    p.state.response_error = error
    with pytest.raises(type(error)) as caught:
        await p.session.cancel_task(accepted.dispatch)
    assert caught.value is error
    retained = await p.store.get(dispatch.dispatch_id)
    assert retained.cancel_state is DispatchState.UNKNOWN
    assert retained.remote_task_id == accepted.response.id
    assert retained.state is DispatchState.ACKNOWLEDGED


async def test_blocking_send_rejects_active_result_without_forgetting_delivery(
    peer_session,
):
    p = peer_session
    request = request_for(p)
    request.configuration.return_immediately = False
    p.state.early_task_response = True
    dispatch, wire = await p.session.reserve(
        request, dispatch_tool="ask_agent", verb="message"
    )
    with pytest.raises(NativePeerProtocolError, match="blocking"):
        await p.session.send(dispatch, wire)
    retained = await p.store.get(dispatch.dispatch_id)
    assert retained.state is DispatchState.ACKNOWLEDGED
    assert retained.remote_task_id and retained.remote_task_state == "submitted"
    assert p.state.received_messages == 1


async def test_sdk_interceptor_cannot_rewrite_the_retained_request_before_signing(
    peer_session,
):
    from a2a.client.interceptors import ClientCallInterceptor

    from kestrel_sovereign.a2a.native_sender import NativeSenderProofError

    p = peer_session
    dispatch, wire = await reserve(p)

    class RewriteMessage(ClientCallInterceptor):
        async def before(self, args):
            changed = SendMessageRequest.FromString(args.input.SerializeToString())
            changed.message.parts[0].text = "Different work from the retained intent"
            args.input = changed

        async def after(self, args):
            return None

    p.session._client._interceptors.insert(0, RewriteMessage())
    with pytest.raises(NativeSenderProofError, match="retained native intent"):
        await p.session.send(dispatch, wire)
    current = await p.store.get(dispatch.dispatch_id)
    assert (
        current.state is DispatchState.NOT_SENT
        and current.request_digest == dispatch.request_digest
    )
    assert not p.trace and not p.state.received_messages


async def test_native_subscription_uses_official_events_and_retires_borrowed_resources(
    network_peer,
):
    p = network_peer
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    stream = p.session.subscribe(accepted.dispatch)
    try:
        first = await asyncio.wait_for(anext(stream), 5)
        assert first.HasField("task") and first.task.id == accepted.response.id
        assert p.admission.manager._subscribers
        await p.admission.manager.complete_task(
            accepted.response.id,
            "Final native subscription result",
            recipient_agent_id=p.admission.scope.recipient_id,
        )
        events = [event async for event in stream]
        assert events[-1].status_update.status.state == TaskState.TASK_STATE_COMPLETED
        assert events[-1].status_update.task_id == accepted.response.id
        assert not p.admission.manager._subscribers
        assert not p.http.is_closed
        current = await p.session.get_task(accepted.dispatch)
        assert current.dispatch.remote_task_state == "completed"
    finally:
        await stream.aclose()


async def test_native_subscription_close_releases_recipient_subscription(network_peer):
    p = network_peer
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    stream = p.session.subscribe(accepted.dispatch)
    first = await asyncio.wait_for(anext(stream), 5)
    assert first.task.id == accepted.response.id
    await stream.aclose()
    async with asyncio.timeout(5):
        while p.admission.manager._subscribers:
            await asyncio.sleep(0.001)
    assert not p.admission.manager._subscribers and not p.http.is_closed


async def test_native_subscription_rechecks_authority_before_delivering_next_event(
    network_peer,
):
    p = network_peer
    dispatch, wire = await reserve(p)
    accepted = await p.session.send(dispatch, wire)
    stream = p.session.subscribe(accepted.dispatch)
    try:
        await asyncio.wait_for(anext(stream), 5)
        p.state.authorized = False
        await p.admission.manager.complete_task(
            accepted.response.id,
            "Must not be exposed after revocation",
            recipient_agent_id=p.admission.scope.recipient_id,
        )
        with pytest.raises(PermissionError, match="revoked"):
            await asyncio.wait_for(anext(stream), 5)
        assert not p.admission.manager._subscribers
        assert (await p.store.get(dispatch.dispatch_id)) == accepted.dispatch
    finally:
        await stream.aclose()


async def test_official_client_commits_taskless_question_answer_with_its_receipt(
    peer_session,
):
    from datetime import UTC, datetime, timedelta
    from kestrel_sovereign.a2a.extensions import (
        NATIVE_EXTENSION_URIS,
        question_correlation_metadata,
    )
    from kestrel_sovereign.a2a.stores.unified.question_store import NativeQuestionStore

    p = peer_session
    questions = NativeQuestionStore(p.store)
    await questions.initialize()
    extensions = frozenset(NATIVE_EXTENSION_URIS)
    original_signer = p.arguments["signer"]
    signer = replace(original_signer, extensions=extensions, nonce=lambda: uuid4().hex)
    session = NativePeerSession(
        **(
            p.arguments
            | {
                "signer": signer,
                "question_store": questions,
            }
        )
    )
    p.state.taskless = True
    question_id = uuid4().hex
    try:
        dispatch, wire = await session.reserve(
            SendMessageRequest(
                tenant=p.admission.scope.tenant,
                message=Message(
                    message_id=uuid4().hex,
                    role=Role.ROLE_USER,
                    parts=[Part(text="A taskless native question")],
                ),
                metadata=question_correlation_metadata("question", question_id),
                configuration={"return_immediately": True},
            ),
            dispatch_tool="send_a2a_question",
            verb="question",
        )
        await questions.reserve(
            dispatch,
            wire,
            origin_turn_id="turn",
            origin_session_id="chat",
            deadline=datetime.now(UTC) + timedelta(seconds=60),
        )
        receipt = await session.send(dispatch, wire)
        retained = await questions.get(question_id)
        assert retained.status == "RESOLVED" and retained.receiver_settled
        assert (
            receipt.dispatch.remote_task_id is None
            and receipt.dispatch.response_kind == "message"
        )
        assert await questions.retained_response(receipt.dispatch) == receipt.response
        [wake] = await questions.pending_wakes()
        assert (
            json.loads(wake.payload_json)["reply_text"] == "Standalone native response"
        )
        assert len(p.trace) == 1
    finally:
        await session.close()
