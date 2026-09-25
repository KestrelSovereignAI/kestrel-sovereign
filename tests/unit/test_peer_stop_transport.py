"""Authenticated peer Stop routing adapters (#3169)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import CausationFrame, SignalMode, SignalResult, Status
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerIdentity,
    PeerRequester,
)
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals.sources.peer_stop import (
    PEER_STOP_RATE_LIMIT_BURST,
    SOURCE_NAME,
    attach_stop_evidence,
    decode_peer_stop_intent,
    dispatch_peer_stop,
    encode_peer_stop_intent,
    peer_stop_delivery_id,
    peer_stop_operation_id,
)
from kestrel_sovereign.stop import (
    StopCleanupRegistry,
    StopDisposition,
    StopOutcome,
    StopScope,
)
# The real dispatcher + receipt-store rail, shared as the ``rail`` fixture.
from tests.unit.test_peer_stop_signals import rail  # noqa: F401


def _payload(
    *,
    correlation_id: str = "peer-stop-transport",
    metadata: dict | None = None,
) -> dict:
    message = encode_peer_stop_intent(
        scope=StopScope.AGENT,
        target=None,
        reason="andon cord",
        cascade=False,
        correlation_id=correlation_id,
    )
    return {
        "id": correlation_id,
        "sessionId": f"session-{correlation_id}",
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": message}],
        },
        "metadata": metadata
        or {
            "sender": "did:test:sender",
            "a2a_verb": "peer_stop",
            "a2a_audience": "did:test:recipient",
        },
    }


def _outcome(
    correlation_id: str,
    *,
    agent_id: str = "did:test:recipient",
    disposition: StopDisposition = StopDisposition.STOPPED,
) -> StopOutcome:
    return StopOutcome(
        scope=StopScope.AGENT,
        requested_target=agent_id,
        resolved_target=agent_id,
        agent_id=agent_id,
        disposition=disposition,
        correlation_id=correlation_id,
    )


def _response(payload: dict, outcomes: list[dict], *, status: str = "ok") -> dict:
    """The recipient's response shape: its receipt correlation is actor-scoped."""

    return {
        "correlation_id": payload["id"],
        "receipt_kind": "operation",
        "stop_correlation_id": outcomes[0]["correlation_id"] if outcomes else "",
        "signal_receipt": {"signal_id": "peer-signal", "status": status, "detail": None},
        "stop_outcomes": outcomes,
    }


def _feature_with_stop_router(*, stop_peer):
    agent = SimpleNamespace(
        did="did:test:sender",
        _agent_name="Sender",
        identity=None,
        _provide_causation_chain=lambda: None,
    )
    feature = PeersFeature(agent)
    feature._own_name = "Sender"
    requester = PeerRequester(agent.did, object())
    peer = PeerIdentity(
        agent_id="did:test:recipient",
        slug="recipient",
        routing_key="recipient-route",
        name="Recipient",
    )
    feature._resolve_automatic_peer = AsyncMock(
        return_value=(SimpleNamespace(stop_peer=stop_peer), requester, peer)
    )
    return feature, peer


@pytest.mark.asyncio
async def test_local_router_reauthorizes_and_posts_peer_stop_route() -> None:
    directory_response = MagicMock(status_code=200)
    directory_response.raise_for_status.return_value = None
    directory_response.json.return_value = [
        {
            "id": "did:test:recipient",
            "name": "Recipient",
            "routing_name": "recipient-route",
        }
    ]
    stop_response = MagicMock(status_code=200)
    stop_response.raise_for_status.return_value = None
    stop_response.json.return_value = {
        "signal_receipt": {
            "signal_id": "signal-1",
            "status": "ok",
            "detail": None,
        },
        "stop_outcomes": [_outcome("peer-stop-transport").to_dict()],
    }
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get.return_value = directory_response
    client.post.return_value = stop_response
    router = LocalHostPeerDirectory(
        "http://local-host",
        client_factory=lambda *args, **kwargs: client,
    )
    requester = PeerRequester("did:test:sender", object())
    peer = PeerIdentity(
        agent_id="did:test:recipient",
        slug="recipient",
        routing_key="recipient-route",
    )
    payload = _payload()

    response = await router.stop_peer(requester, peer, payload)

    assert response["signal_receipt"]["status"] == "ok"
    post = client.post.await_args
    assert post.args[0] == (
        "http://local-host/api/agents/recipient-route/api/agent/peer/stop"
    )
    assert post.kwargs["json"] == payload


@pytest.mark.asyncio
async def test_local_router_uses_host_capability_without_http() -> None:
    local_stop = AsyncMock(
        return_value={
            "signal_receipt": {
                "signal_id": "signal-local",
                "status": "ok",
                "detail": None,
            },
            "stop_outcomes": [_outcome("peer-stop-transport").to_dict()],
        }
    )
    client_factory = MagicMock()
    router = LocalHostPeerDirectory(
        "http://local-host",
        client_factory=client_factory,
        local_stop=local_stop,
    )
    router._directory_entries = AsyncMock(
        return_value=[
            PeerIdentity(
                agent_id="did:test:recipient",
                slug="recipient",
                routing_key="recipient-route",
            )
        ]
    )
    requester = PeerRequester("did:test:sender", object())
    peer = PeerIdentity(
        agent_id="did:test:recipient",
        slug="recipient",
        routing_key="recipient-route",
    )
    payload = _payload()

    response = await router.stop_peer(requester, peer, payload)

    assert response["signal_receipt"]["status"] == "ok"
    local_stop.assert_awaited_once_with(requester, peer, payload)
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_peer_tool_routes_without_serializing_principals() -> None:
    recipient_id = "did:test:recipient"
    agent = SimpleNamespace(
        did="did:test:sender",
        _agent_name="Sender",
        identity=None,
        _provide_causation_chain=lambda: None,
    )
    feature = PeersFeature(agent)
    feature._own_name = "Sender"
    requester = PeerRequester(agent.did, object())
    peer = PeerIdentity(
        agent_id=recipient_id,
        slug="recipient",
        routing_key="recipient-route",
        name="Recipient",
    )
    captured: list[dict] = []

    async def stop_peer(_requester, _peer, payload):
        captured.append(payload)
        operation_id = peer_stop_operation_id(agent.did, payload["id"])
        return _response(
            payload, [_outcome(operation_id, agent_id=recipient_id).to_dict()]
        )

    router = SimpleNamespace(stop_peer=stop_peer)
    feature._resolve_automatic_peer = AsyncMock(
        return_value=(router, requester, peer)
    )

    result = await feature.stop_peer("Recipient", reason="unsafe loop")

    assert result.status is ToolResultStatus.OK
    assert result.data["stopped"] is True
    assert result.data["recipient_agent_id"] == recipient_id
    assert len(captured) == 1
    wire = captured[0]
    assert "actor_id" not in wire["message"]["parts"][0]["text"]
    assert "target_agent_id" not in wire["message"]["parts"][0]["text"]
    assert recipient_id not in wire["message"]["parts"][0]["text"]
    assert '"cascade":false' in wire["message"]["parts"][0]["text"]
    assert wire["metadata"]["a2a_audience"] == recipient_id


@pytest.mark.asyncio
async def test_peer_tool_rejects_unimplemented_tool_call_scope() -> None:
    feature = PeersFeature(
        SimpleNamespace(
            did="did:test:sender",
            _agent_name="Sender",
            identity=None,
        )
    )
    feature._resolve_automatic_peer = AsyncMock()

    result = await feature.stop_peer(
        "Recipient",
        scope="tool_call",
        target="tool-call-1",
    )

    assert result.status is ToolResultStatus.ERROR
    assert "cannot target tool_call scope" in result.error
    feature._resolve_automatic_peer.assert_not_awaited()


@pytest.mark.asyncio
async def test_peer_tool_converts_unexpected_router_failure() -> None:
    feature, _peer = _feature_with_stop_router(
        stop_peer=AsyncMock(side_effect=RuntimeError("provider detail"))
    )

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop dispatch failed"
    assert "provider detail" not in str(result.data)


@pytest.mark.asyncio
async def test_peer_tool_reports_refused_outcome_as_failure() -> None:
    # A handler-decided refusal is keyed by the Stop operation.
    async def stop_peer(_requester, peer, payload):
        outcome = _outcome(
            peer_stop_operation_id("did:test:sender", payload["id"]),
            agent_id=peer.agent_id,
            disposition=StopDisposition.REFUSED,
        )
        return _response(payload, [outcome.to_dict()])

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop was refused or could not be confirmed"
    assert result.data["stopped"] is False


def _delivery_response(payload: dict, outcome: StopOutcome, *, signal_id: str) -> dict:
    """A dispatcher refusal, in ``dispatch_peer_stop``'s response shape.

    The response names the record its outcome is, as the recipient does;
    the sender must still bind that record to the signal that answered.
    """

    return {
        "correlation_id": payload["id"],
        "receipt_kind": "delivery",
        "stop_correlation_id": outcome.correlation_id,
        "signal_receipt": {
            "signal_id": signal_id,
            "status": "dropped_rate_limit",
            "detail": "Peer Stop was refused by peer rate limits",
        },
        "stop_outcomes": [outcome.to_dict()],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome_signal", "disposition"),
    [
        # Bound to a different delivery than the one that answered.
        ("other-signal", StopDisposition.REFUSED),
        # A delivery-keyed outcome is a dispatcher decision; never an effect.
        ("peer-signal", StopDisposition.STOPPED),
        ("peer-signal", StopDisposition.ALREADY_COMPLETE),
    ],
)
async def test_peer_tool_binds_a_delivery_outcome_to_its_signal_and_a_refusal(
    outcome_signal, disposition
) -> None:
    async def stop_peer(_requester, peer, payload):
        outcome = _outcome(
            peer_stop_delivery_id(outcome_signal),
            agent_id=peer.agent_id,
            disposition=disposition,
        )
        return _delivery_response(payload, outcome, signal_id="peer-signal")

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop receipt did not match the routed peer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mislabel",
    [
        # A delivery record named by the operation id.
        "delivery_named_as_operation",
        # An operation record whose key is really the delivery's.
        "operation_named_by_delivery",
    ],
)
async def test_peer_tool_matches_each_record_kind_by_its_own_identity(
    mislabel,
) -> None:
    async def stop_peer(_requester, peer, payload):
        delivery = _outcome(
            peer_stop_delivery_id("peer-signal"),
            agent_id=peer.agent_id,
            disposition=StopDisposition.REFUSED,
        )
        response = _delivery_response(payload, delivery, signal_id="peer-signal")
        if mislabel == "delivery_named_as_operation":
            response["stop_correlation_id"] = peer_stop_operation_id(
                "did:test:sender", payload["id"]
            )
        else:
            response["receipt_kind"] = "operation"
        return response

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop receipt did not match the routed peer"


@pytest.mark.asyncio
async def test_peer_tool_accepts_a_delivery_refused_before_dispatch() -> None:
    """Identity reuse is refused before any dispatcher decision (status None)."""

    async def stop_peer(_requester, peer, payload):
        refusal = _outcome(
            peer_stop_delivery_id("never-dispatched"),
            agent_id=peer.agent_id,
            disposition=StopDisposition.REFUSED,
        )
        response = _delivery_response(
            payload, refusal, signal_id="never-dispatched"
        )
        response["signal_receipt"]["status"] = None
        return response

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop was refused or could not be confirmed"
    assert result.data["receipt_kind"] == "delivery"


def _loop_frame(agent_id: str) -> CausationFrame:
    return CausationFrame(
        agent_id=agent_id,
        source=SOURCE_NAME,
        signal_id="earlier",
        turn_id=None,
        depth=1,
        emitted_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refusal", "status", "detail"),
    [
        ("rate_limit", "dropped_rate_limit", "Peer Stop was refused by peer rate limits"),
        (
            "cycle",
            "dropped_cycle",
            "Peer Stop was refused by causation cycle or depth policy",
        ),
    ],
)
async def test_peer_tool_surfaces_a_real_dispatcher_refusal(
    rail,  # noqa: F811 - the imported fixture
    refusal,
    status,
    detail,
) -> None:
    """A refusal the recipient's real dispatcher decided reaches the caller typed."""

    recipient = rail.agent
    recipient._active_request_ids.add("req-live")
    if refusal == "rate_limit":
        for index in range(PEER_STOP_RATE_LIMIT_BURST):
            await dispatch_peer_stop(
                recipient,
                actor_id=f"did:test:other-peer-{index}",
                intent={
                    "scope": "agent",
                    "target": None,
                    "reason": "fill the budget",
                    "cascade": False,
                    "correlation_id": f"fill-{index}",
                },
            )
        recipient.cancelled.clear()
        recipient._active_request_ids.add("req-live")
    chain = [_loop_frame(recipient.did)] if refusal == "cycle" else ()

    async def stop_peer(_requester, _peer, payload):
        assert payload["metadata"]["a2a_audience"] == recipient.did
        intent = decode_peer_stop_intent(payload["message"]["parts"][0]["text"])
        return await dispatch_peer_stop(
            recipient,
            actor_id="did:test:sender",
            intent=intent,
            causation_chain=chain,
        )

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)
    feature._resolve_automatic_peer = AsyncMock(
        return_value=(
            SimpleNamespace(stop_peer=stop_peer),
            PeerRequester("did:test:sender", object()),
            PeerIdentity(
                agent_id=recipient.did,
                slug="recipient",
                routing_key="recipient-route",
                name="Recipient",
            ),
        )
    )

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop was refused or could not be confirmed"
    assert result.data["stopped"] is False
    assert result.data["signal_receipt"]["status"] == status
    assert result.data["signal_receipt"]["detail"] == detail
    [outcome] = result.data["stop_outcomes"]
    assert outcome["disposition"] == "refused"
    assert outcome["detail"] == detail
    assert outcome["correlation_id"] == peer_stop_delivery_id(
        result.data["signal_receipt"]["signal_id"]
    )
    assert recipient.cancelled == []
    # The refusal is receipted under the verified sender, visible in history.
    [refusal_receipt] = [
        receipt
        for receipt in (await rail.receipts.list_receipts(limit=20)).receipts
        if receipt.actor_id == "did:test:sender"
    ]
    assert [o.disposition for o in refusal_receipt.outcomes] == ["refused"]
    assert refusal_receipt.outcomes[0].detail == detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"signal_id": "signal", "status": "unknown", "detail": None},
        {"signal_id": "", "status": "ok", "detail": None},
        {"signal_id": "signal", "status": "ok", "detail": {"raw": "error"}},
        # Only a delivery may lack a dispatcher decision.
        {"signal_id": "signal", "status": None, "detail": None},
    ],
)
async def test_peer_tool_rejects_malformed_signal_receipt(receipt) -> None:
    async def stop_peer(_requester, peer, payload):
        response = _response(
            payload,
            [_outcome(payload["id"], agent_id=peer.agent_id).to_dict()],
        )
        response["signal_receipt"] = receipt
        return response

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer returned a malformed Stop receipt"
    assert result.data["stopped"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome_scope", "requested_target"),
    [
        (StopScope.TURN, "turn-other"),
        (StopScope.AGENT, "did:test:other-target"),
    ],
)
async def test_peer_tool_rejects_outcome_for_different_stop_address(
    outcome_scope: StopScope,
    requested_target: str,
) -> None:
    correlation: list[str] = []

    async def stop_peer(_requester, peer, payload):
        correlation.append(payload["id"])
        outcome = StopOutcome(
            scope=outcome_scope,
            requested_target=requested_target,
            resolved_target=peer.agent_id,
            agent_id=peer.agent_id,
            disposition=StopDisposition.STOPPED,
            correlation_id=payload["id"],
        )
        return _response(payload, [outcome.to_dict()])

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert correlation
    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop receipt did not match the routed peer"


@pytest.mark.asyncio
@pytest.mark.parametrize("malformation", ["duplicate", "foreign_correlation"])
async def test_peer_tool_rejects_noncanonical_outcome_envelope(
    malformation: str,
) -> None:
    async def stop_peer(_requester, peer, payload):
        outcome = _outcome(payload["id"], agent_id=peer.agent_id).to_dict()
        outcomes = [outcome]
        response = _response(payload, outcomes)
        if malformation == "duplicate":
            outcomes.append(dict(outcome))
        else:
            response["correlation_id"] = "someone-elses-request"
        return response

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop receipt did not match the routed peer"
    assert result.data["stopped"] is False


@pytest.mark.asyncio
async def test_peer_tool_accepts_outcome_for_exact_turn_address() -> None:
    async def stop_peer(_requester, peer, payload):
        outcome = StopOutcome(
            scope=StopScope.TURN,
            requested_target="turn-1",
            # The receipt blinds the private request key a public turn
            # resolved to; the tool checks only the address it asked for.
            resolved_target="sha256:blinded-request-key",
            agent_id=peer.agent_id,
            disposition=StopDisposition.STOPPED,
            correlation_id=payload["id"],
        )
        return _response(payload, [outcome.to_dict()])

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer(
        "Recipient",
        scope="turn",
        target="turn-1",
    )

    assert result.status is ToolResultStatus.OK
    assert result.data["stopped"] is True


@pytest.mark.asyncio
async def test_manager_host_attestation_dispatches_signal_with_live_sender_chain() -> None:
    manager = AgentManager()
    sender = SimpleNamespace(
        did="did:test:sender",
        agent_id="did:test:sender",
        _provide_causation_chain=lambda: [
            {
                "agent_id": "did:test:sender",
                "source": "heartbeat",
                "signal_id": "live-chain",
                "turn_id": "turn-live",
                "depth": 1,
                "emitted_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    recipient = SimpleNamespace(
        did="did:test:recipient",
        agent_id="did:test:recipient",
    )
    manager._register_agent("Sender", sender)
    manager._register_agent("Recipient", recipient)
    sender_requester = PeerRequester(sender.did, object())
    recipient_requester = PeerRequester(recipient.did, object())
    sender_router = SimpleNamespace()
    recipient_router = SimpleNamespace(
        authorize_inbound_sender=AsyncMock(return_value=True)
    )
    manager.install_a2a_hosted_policy(
        sender,
        resolver=None,
        authorizer=None,
        router=sender_router,
        requester=sender_requester,
    )
    manager.install_a2a_hosted_policy(
        recipient,
        resolver=None,
        authorizer=None,
        router=recipient_router,
        requester=recipient_requester,
    )
    peer = PeerIdentity(
        agent_id=recipient.did,
        slug="recipient",
        routing_key="Recipient",
    )

    async def dispatch(signal, *, source_event_id=None):
        assert signal.caller == sender.did
        assert signal.target_agent == recipient.did
        assert [frame.signal_id for frame in signal.causation_chain] == [
            "live-chain"
        ]
        assert source_event_id == peer_stop_operation_id(
            sender.did, signal.payload["correlation_id"]
        )
        outcome = _outcome(source_event_id)
        return SignalResult(
            signal_id=signal.id,
            status=Status.OK,
            mode=SignalMode.ACTION,
            duration_ms=1,
            action_result=[outcome.to_dict()],
        )

    recipient.dispatcher = SimpleNamespace(
        dispatch_signal=AsyncMock(side_effect=dispatch)
    )
    receipts = SimpleNamespace(bind_operation=AsyncMock(return_value=None))
    attach_stop_evidence(
        recipient, receipt_store=receipts, cleanup_registry=StopCleanupRegistry()
    )
    payload = _payload(
        metadata={
            "sender": "did:test:forged",
            "a2a_verb": "peer_stop",
            "a2a_audience": recipient.did,
            "causation_chain": [
                {
                    "agent_id": "did:test:forged",
                    "source": "forged",
                    "signal_id": "forged-chain",
                    "turn_id": None,
                    "depth": 1,
                    "emitted_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }
    )

    response = await manager.stop_host_attested_local_peer(
        sender=sender,
        requester=sender_requester,
        peer=peer,
        payload=payload,
    )

    assert response["signal_receipt"]["status"] == "ok"
    recipient.dispatcher.dispatch_signal.assert_awaited_once()
    # The operation was bound under the attested sender before dispatch.
    [bound] = receipts.bind_operation.await_args.args
    assert bound.actor_id == sender.did
    recipient_router.authorize_inbound_sender.assert_awaited_once_with(
        recipient_requester,
        sender.did,
    )


@pytest.mark.asyncio
async def test_manager_rejects_peer_stop_retargeted_after_route_authorization() -> None:
    manager = AgentManager()
    sender = SimpleNamespace(did="did:test:sender")
    recipient = SimpleNamespace(
        did="did:test:actual-recipient",
        dispatcher=SimpleNamespace(dispatch_signal=AsyncMock()),
    )
    manager._authorize_host_attested_local_a2a_route = AsyncMock(
        return_value=(sender.did, recipient)
    )
    payload = _payload(
        metadata={
            "sender": sender.did,
            "a2a_verb": "peer_stop",
            "a2a_audience": "did:test:intended-recipient",
        }
    )

    with pytest.raises(PeerAccessDeniedError, match="audience"):
        await manager.stop_host_attested_local_peer(
            sender=sender,
            requester=object(),
            peer=object(),
            payload=payload,
        )

    recipient.dispatcher.dispatch_signal.assert_not_awaited()


# ---------------------------------------------------------------------------
# Lost response: the sender re-signs, the wire door answers from the receipt
# ---------------------------------------------------------------------------

_WIRE_SENDER_DID = "did:web:example.com:agent:stop-sender"


def _hybrid_sender_identity():
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair

    keypair = generate_hybrid_keypair()
    methods = build_verification_methods(_WIRE_SENDER_DID, keypair.public_keys())
    identity = SimpleNamespace(
        is_hybrid=True,
        hybrid_keypair=keypair,
        signing_did=_WIRE_SENDER_DID,
        new_verification_methods=list(methods),
    )
    return identity, {"id": _WIRE_SENDER_DID, "verificationMethod": list(methods)}


def _peer_stop_wire_app(recipient):
    from fastapi import FastAPI
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    from kestrel_sovereign.endpoints import agent as agent_endpoint

    app = FastAPI()
    app.state.limiter = Limiter(key_func=get_remote_address)
    app.include_router(agent_endpoint.router)

    @app.middleware("http")
    async def _attach_recipient(request, call_next):
        request.state.agent = recipient
        return await call_next(request)

    return app


def _lossy_host_transport(app, *, peer_route: str, peer_did: str):
    """A local host whose first peer Stop response never reaches the sender.

    The request itself is delivered to the real endpoint, so the recipient
    dispatches and receipts the Stop before the connection drops.
    """

    import httpx

    wire = SimpleNamespace(bodies=[], statuses=[])
    route_prefix = f"/api/agents/{peer_route}"
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://recipient"
    )

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/agents":
            return httpx.Response(
                200,
                json=[{"id": peer_did, "name": "Recipient", "routing_name": peer_route}],
            )
        assert request.url.path.startswith(route_prefix)
        wire.bodies.append(request.content)
        forwarded = await upstream.post(
            request.url.path[len(route_prefix):],
            content=request.content,
            headers={"Content-Type": "application/json"},
        )
        wire.statuses.append(forwarded.status_code)
        if len(wire.bodies) == 1:
            raise httpx.ReadTimeout("response lost after delivery", request=request)
        return httpx.Response(
            forwarded.status_code,
            content=forwarded.content,
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handle), upstream, wire


@pytest.mark.asyncio
async def test_lost_wire_response_is_retried_and_answered_from_the_original_receipt(
    rail,  # noqa: F811 - the imported fixture
) -> None:
    import json

    import httpx

    from kestrel_sdk.signals import Status as SignalStatus

    identity, sender_document = _hybrid_sender_identity()
    recipient = rail.agent
    recipient.a2a_did_resolver = (
        lambda did: sender_document if did == _WIRE_SENDER_DID else None
    )
    recipient.a2a_inbound_sender_authorizer = None
    recipient.peer_directory_router = None
    recipient.peer_requester = None
    recipient._a2a_host_manager = None
    recipient._active_request_ids.add("req-live")
    app = _peer_stop_wire_app(recipient)
    transport, upstream, wire = _lossy_host_transport(
        app, peer_route="recipient-route", peer_did=recipient.did
    )

    sender_agent = SimpleNamespace(
        did="did:pkh:sender",
        _agent_name="Sender",
        identity=identity,
        _provide_causation_chain=lambda: None,
    )
    feature = PeersFeature(sender_agent)
    feature._own_name = "Sender"
    router = LocalHostPeerDirectory(
        "http://local-host",
        client_factory=lambda *args, **kwargs: httpx.AsyncClient(
            transport=transport
        ),
    )
    requester = PeerRequester(sender_agent.did, object())
    peer = PeerIdentity(
        agent_id=recipient.did,
        slug="recipient",
        routing_key="recipient-route",
        name="Recipient",
    )
    feature._resolve_automatic_peer = AsyncMock(return_value=(router, requester, peer))

    try:
        result = await feature.stop_peer("recipient")

        # The lost first delivery DID stop the peer; the retry was a new,
        # validly signed request for the same Stop, not a verbatim replay.
        assert wire.statuses == [200, 200]
        first, second = (json.loads(body) for body in wire.bodies)
        assert first["id"] == second["id"]
        assert first["message"] == second["message"]
        assert (
            first["metadata"]["signature"]["nonce"]
            != second["metadata"]["signature"]["nonce"]
        )

        assert result.status is ToolResultStatus.OK
        assert result.data["signal_receipt"]["status"] == SignalStatus.COALESCED.value
        [outcome] = result.data["stop_outcomes"]
        assert outcome["disposition"] == "stopped"
        assert result.data["stopped"] is True
        # One effect, one durable receipt, attributed to the verified sender.
        assert recipient.cancelled == ["req-live"]
        [receipt] = (await rail.receipts.list_receipts(limit=10)).receipts
        assert receipt.actor_id == _WIRE_SENDER_DID
        [recorded] = receipt.outcomes
        assert recorded.disposition == outcome["disposition"]
        assert outcome["agent_id"] == recipient.did

        # A byte-identical resend is still refused by the replay nonce; the
        # retry contract is re-signing, not replaying.
        verbatim = await upstream.post(
            "/api/agent/peer/stop",
            content=wire.bodies[1],
            headers={"Content-Type": "application/json"},
        )
        assert verbatim.status_code == 403
        assert recipient.cancelled == ["req-live"]
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_peer_tool_does_not_retry_an_authorization_decision() -> None:
    stop_peer = AsyncMock(side_effect=PeerAccessDeniedError("denied"))
    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("recipient")

    assert result.status is ToolResultStatus.ERROR
    stop_peer.assert_awaited_once()


@pytest.mark.asyncio
async def test_peer_tool_bounds_retries_of_an_unconfirmed_delivery() -> None:
    from kestrel_sovereign.features.peers.directory import PeerTransportError
    from kestrel_sovereign.signals.sources.peer_stop import (
        PEER_STOP_DELIVERY_ATTEMPTS,
    )

    stop_peer = AsyncMock(side_effect=PeerTransportError("lost"))
    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["stopped"] is False
    assert stop_peer.await_count == PEER_STOP_DELIVERY_ATTEMPTS
    correlation_ids = {call.args[2]["id"] for call in stop_peer.await_args_list}
    assert len(correlation_ids) == 1
