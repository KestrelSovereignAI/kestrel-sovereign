"""Authenticated peer Stop routing adapters (#3169)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import SignalMode, SignalResult, Status
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerIdentity,
    PeerRequester,
)
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.signals.sources.peer_stop import encode_peer_stop_intent
from kestrel_sovereign.stop import StopDisposition, StopOutcome, StopScope


def _payload(
    *,
    correlation_id: str = "peer-stop-transport",
    metadata: dict | None = None,
) -> dict:
    message = encode_peer_stop_intent(
        scope=StopScope.AGENT,
        target=None,
        reason="andon cord",
        cascade=True,
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
        correlation_id = payload["id"]
        return {
            "signal_receipt": {
                "signal_id": "peer-signal",
                "status": "ok",
                "detail": None,
            },
            "stop_outcomes": [
                _outcome(correlation_id, agent_id=recipient_id).to_dict()
            ],
        }

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
async def test_peer_tool_rejects_stopped_outcome_with_failed_signal_receipt() -> None:
    async def stop_peer(_requester, peer, payload):
        return {
            "signal_receipt": {
                "signal_id": "failed-signal",
                "status": "failed",
                "detail": "Peer Stop signal failed",
            },
            "stop_outcomes": [
                _outcome(payload["id"], agent_id=peer.agent_id).to_dict()
            ],
        }

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop was refused or could not be confirmed"
    assert result.data["stopped"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"signal_id": "signal", "status": "unknown", "detail": None},
        {"signal_id": "", "status": "ok", "detail": None},
        {"signal_id": "signal", "status": "ok", "detail": {"raw": "error"}},
    ],
)
async def test_peer_tool_rejects_malformed_signal_receipt(receipt) -> None:
    async def stop_peer(_requester, peer, payload):
        return {
            "signal_receipt": receipt,
            "stop_outcomes": [
                _outcome(payload["id"], agent_id=peer.agent_id).to_dict()
            ],
        }

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
        return {
            "signal_receipt": {
                "signal_id": "mismatched-signal",
                "status": "ok",
                "detail": None,
            },
            "stop_outcomes": [outcome.to_dict()],
        }

    feature, _peer = _feature_with_stop_router(stop_peer=stop_peer)

    result = await feature.stop_peer("Recipient")

    assert correlation
    assert result.status is ToolResultStatus.ERROR
    assert result.error == "Peer Stop receipt did not match the routed peer"


@pytest.mark.asyncio
@pytest.mark.parametrize("malformation", ["duplicate", "resolved_target"])
async def test_peer_tool_rejects_noncanonical_outcome_envelope(
    malformation: str,
) -> None:
    async def stop_peer(_requester, peer, payload):
        outcome = _outcome(payload["id"], agent_id=peer.agent_id).to_dict()
        outcomes = [outcome]
        if malformation == "duplicate":
            outcomes.append(dict(outcome))
        else:
            outcome["resolved_target"] = "did:test:other-runtime"
        return {
            "signal_receipt": {
                "signal_id": "noncanonical-signal",
                "status": "ok",
                "detail": None,
            },
            "stop_outcomes": outcomes,
        }

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
            # The remote handler resolves a private request-generation key,
            # but the peer-facing receipt redacts that address to the routed
            # authenticated agent identity.
            resolved_target=peer.agent_id,
            agent_id=peer.agent_id,
            disposition=StopDisposition.STOPPED,
            correlation_id=payload["id"],
        )
        return {
            "signal_receipt": {
                "signal_id": "matched-signal",
                "status": "ok",
                "detail": None,
            },
            "stop_outcomes": [outcome.to_dict()],
        }

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
        assert source_event_id == signal.dedupe_key
        outcome = _outcome(signal.payload["correlation_id"])
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
