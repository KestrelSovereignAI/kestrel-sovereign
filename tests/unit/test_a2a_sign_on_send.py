"""A2A sign-on-send (#1706).

A hybrid agent signs its outbound task envelope so the recipient can verify it
(#1673/#1705). Tests use real keypairs and check the produced signature
verifies end-to-end; non-hybrid agents send unsigned (back-compat).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.a2a.envelope_signing import (
    canonical_message,
    kids_from_verification_methods,
    verify_envelope,
    verify_inbound_envelope,
)
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.peers.feature import (
    OutboundSigningError,
    PeersFeature,
)


DID = "did:web:example.com:agent:emma"


def _hybrid_identity():
    kp = generate_hybrid_keypair()
    vms = build_verification_methods(DID, kp.public_keys())
    identity = SimpleNamespace(
        is_hybrid=True, hybrid_keypair=kp, signing_did=DID, new_verification_methods=vms
    )
    return identity, kp, vms


def _feature(identity=None):
    agent = SimpleNamespace(_agent_name="emma", identity=identity)
    return PeersFeature(agent)


def _payload():
    return {
        "id": "t1",
        "sessionId": "s1",
        "message": {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
        "metadata": {"sender": "emma"},
    }


# --------------------------------------------------------------------------
# kid derivation
# --------------------------------------------------------------------------


def test_kids_derived_from_vms():
    _, _, vms = _hybrid_identity()
    classical, pq = kids_from_verification_methods(vms)
    # build_verification_methods uses #key-1 / #key-2 (the ceremony default).
    assert classical == "key-1" and pq == "key-2"


def test_kids_fallback_when_vms_missing():
    assert kids_from_verification_methods([]) == ("key-1", "key-2")
    assert kids_from_verification_methods([{"id": "did:x#ed25519"}]) == ("ed25519", "key-2")


# --------------------------------------------------------------------------
# _maybe_sign_outbound
# --------------------------------------------------------------------------


def test_hybrid_agent_signs_and_sets_did_sender():
    identity, kp, vms = _hybrid_identity()
    feature = _feature(identity)
    payload = _payload()

    feature._maybe_sign_outbound(payload, task_id="t1", sess_id="s1", message="hello")

    md = payload["metadata"]
    assert md["sender"] == DID  # verified identifier, not the display name
    assert "signature" in md
    # The produced signature verifies against the agent's own VMs. v2 binds the
    # nonce + behaviour-steering fields, so reconstruct them as the verifier does.
    from kestrel_sovereign.a2a.envelope_signing import bound_envelope_fields

    block = md["signature"]
    assert block["v"] == 2 and block["nonce"]
    bound = bound_envelope_fields(md, artifacts=payload.get("artifacts"))
    doc = {"id": DID, "verificationMethod": vms}
    v = verify_envelope(
        doc, block, sender=DID, task_id="t1",
        message=canonical_message(["hello"]), session_id="s1",
        timestamp=block["timestamp"], bound=bound, nonce=block["nonce"],
    )
    assert v.ok is True and v.verified is True


def test_non_hybrid_agent_sends_unsigned():
    identity = SimpleNamespace(is_hybrid=False)
    feature = _feature(identity)
    payload = _payload()

    feature._maybe_sign_outbound(payload, task_id="t1", sess_id="s1", message="hello")

    assert "signature" not in payload["metadata"]
    assert payload["metadata"]["sender"] == "emma"  # unchanged


def test_no_identity_sends_unsigned():
    feature = _feature(identity=None)
    payload = _payload()
    feature._maybe_sign_outbound(payload, task_id="t1", sess_id="s1", message="hello")
    assert "signature" not in payload["metadata"]


def test_loaded_hybrid_identity_missing_material_fails_closed():
    identity = SimpleNamespace(
        is_hybrid=True, hybrid_keypair=None, signing_did=DID, new_verification_methods=[{"id": f"{DID}#key-1"}]
    )
    feature = _feature(identity)
    payload = _payload()

    with pytest.raises(OutboundSigningError) as exc_info:
        feature._maybe_sign_outbound(
            payload, task_id="t1", sess_id="s1", message="hello",
        )

    assert exc_info.value.code == "missing_hybrid_signing_material"
    assert "signature" not in payload["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("send_a2a_message", {}),
        ("send_a2a_question", {}),
        ("send_a2a_task", {"skill_id": "workflow.assign"}),
    ],
)
async def test_hybrid_signer_exception_aborts_every_dispatch_without_http(
    method_name,
    kwargs,
):
    """Every public A2A verb shares the same fail-closed signing boundary."""
    identity, _, _ = _hybrid_identity()
    feature = _feature(identity)
    feature._host_url = "http://multi-agent"
    feature._api_key = ""
    feature._own_name = "emma"
    client_factory = MagicMock()

    with (
        patch(
            "kestrel_sovereign.a2a.envelope_signing.sign_envelope",
            side_effect=RuntimeError("provider detail must stay private"),
        ),
        patch(
            "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
            client_factory,
        ),
    ):
        result = await getattr(feature, method_name)(
            "meridian", "authenticated handoff", **kwargs,
        )

    assert result.status is ToolResultStatus.ERROR
    assert result.data["sent"] is False
    assert result.data["error_type"] == "a2a_signing_failed"
    assert result.data["error_code"] == "hybrid_signer_error"
    assert "signing failed" in (result.error or "").lower()
    assert "provider detail" not in (result.error or "")
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_signing_failure_cannot_retry_as_unsigned():
    """Repeated caller attempts must re-sign and must never reach HTTP."""
    identity, _, _ = _hybrid_identity()
    feature = _feature(identity)
    feature._host_url = "http://multi-agent"
    feature._api_key = ""
    feature._own_name = "emma"
    client_factory = MagicMock()

    with (
        patch(
            "kestrel_sovereign.a2a.envelope_signing.sign_envelope",
            new=MagicMock(side_effect=RuntimeError("signer unavailable")),
        ) as signer,
        patch(
            "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
            client_factory,
        ),
    ):
        first = await feature.send_a2a_task("meridian", "one")
        second = await feature.send_a2a_task("meridian", "two")

    assert first.data["error_type"] == "a2a_signing_failed"
    assert second.data["error_type"] == "a2a_signing_failed"
    assert signer.call_count == 2
    client_factory.assert_not_called()


@pytest.mark.asyncio
async def test_public_hybrid_dispatch_posts_envelope_verifying_both_halves():
    """The real send path emits Ed25519 + ML-DSA and passes hybrid policy."""
    identity, _, vms = _hybrid_identity()
    feature = _feature(identity)
    feature._host_url = "http://multi-agent"
    feature._api_key = ""
    feature._own_name = "emma"
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": {"state": "submitted"},
    }
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = response

    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        result = await feature.send_a2a_task(
            "meridian", "authenticated handoff", skill_id="workflow.assign",
        )

    assert result.status is ToolResultStatus.OK
    posted = client.post.call_args.kwargs["json"]
    signatures = posted["metadata"]["signature"]["signatures"]
    assert {entry["alg"] for entry in signatures} == {"ed25519", "ml-dsa-65"}
    verdict = await verify_inbound_envelope(
        posted["metadata"],
        task_id=posted["id"],
        message=canonical_message(["authenticated handoff"]),
        session_id=posted["sessionId"],
        artifacts=posted.get("artifacts"),
        resolver=lambda did: (
            {"id": DID, "verificationMethod": vms} if did == DID else None
        ),
    )
    assert verdict.ok is True and verdict.verified is True
    assert verdict.sender == DID


# --------------------------------------------------------------------------
# end-to-end: signed-on-send verifies through the inbound decision function
# --------------------------------------------------------------------------


def test_signed_on_send_verifies_inbound_end_to_end():
    identity, kp, vms = _hybrid_identity()
    feature = _feature(identity)
    payload = _payload()
    feature._maybe_sign_outbound(payload, task_id="t1", sess_id="s1", message="hello")

    # Recipient side: resolve emma's DID to its doc, verify the inbound envelope
    # exactly as /tasks/send does (canonical message + authoritative sessionId).
    doc = {"id": DID, "verificationMethod": vms}
    verdict = asyncio.run(verify_inbound_envelope(
        payload["metadata"],
        task_id="t1",
        message=canonical_message(["hello"]),
        session_id="s1",
        resolver=lambda did: doc if did == DID else None,
    ))
    assert verdict.ok is True and verdict.verified is True


def test_signed_on_send_rejected_if_message_differs():
    identity, kp, vms = _hybrid_identity()
    feature = _feature(identity)
    payload = _payload()
    feature._maybe_sign_outbound(payload, task_id="t1", sess_id="s1", message="hello")

    doc = {"id": DID, "verificationMethod": vms}
    verdict = asyncio.run(verify_inbound_envelope(
        payload["metadata"], task_id="t1",
        message=canonical_message(["DIFFERENT"]), session_id="s1",
        resolver=lambda did: doc,
    ))
    assert verdict.ok is False
