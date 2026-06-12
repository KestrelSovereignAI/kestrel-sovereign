"""A2A sign-on-send (#1706).

A hybrid agent signs its outbound task envelope so the recipient can verify it
(#1673/#1705). Tests use real keypairs and check the produced signature
verifies end-to-end; non-hybrid agents send unsigned (back-compat).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.a2a.envelope_signing import (
    canonical_message,
    kids_from_verification_methods,
    verify_envelope,
    verify_inbound_envelope,
)
from kestrel_sovereign.features.peers.feature import PeersFeature


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


def test_signing_failure_falls_back_to_unsigned():
    # is_hybrid True but keypair missing -> guarded, no signature, no raise.
    identity = SimpleNamespace(
        is_hybrid=True, hybrid_keypair=None, signing_did=DID, new_verification_methods=[{"id": f"{DID}#key-1"}]
    )
    feature = _feature(identity)
    payload = _payload()
    feature._maybe_sign_outbound(payload, task_id="t1", sess_id="s1", message="hello")
    assert "signature" not in payload["metadata"]


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
