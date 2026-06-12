"""A2A signed-sender envelope contract (#1673).

Cryptographic sender authentication for A2A: a sender signs a canonical view of
the envelope with its hybrid (Ed25519 + ML-DSA-65) key; a receiver verifies it
against the sender's DID document. These tests pin the full round-trip plus the
tamper / replay / binding rejections and the inbound back-compat decision
matrix — all with real keys, no mocks of the crypto.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from kestrel_sovereign.identity.hybrid_keypair import generate_hybrid_keypair
from kestrel_sovereign.identity.did_web import build_verification_methods
from kestrel_sovereign.security.verify_policy import VerifyPolicy
from kestrel_sovereign.a2a.envelope_signing import (
    canonical_signing_bytes,
    sign_envelope,
    verify_envelope,
    verify_inbound_envelope,
)


DID = "did:web:example.com:agent:meridian"


def _keypair_and_doc(did: str = DID):
    kp = generate_hybrid_keypair()
    doc = {
        "id": did,
        "verificationMethod": build_verification_methods(did, kp.public_keys()),
    }
    return kp, doc


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


# --------------------------------------------------------------------------
# canonical bytes
# --------------------------------------------------------------------------


def test_canonical_bytes_are_deterministic_and_order_independent():
    a = canonical_signing_bytes(sender="s", task_id="t", session_id="x", message="hi", timestamp="2026-01-01T00:00:00+00:00")
    b = canonical_signing_bytes(message="hi", timestamp="2026-01-01T00:00:00+00:00", task_id="t", sender="s", session_id="x")
    assert a == b
    # None session id normalises to "" (same bytes as explicit "").
    c = canonical_signing_bytes(sender="s", task_id="t", session_id=None, message="hi", timestamp="2026-01-01T00:00:00+00:00")
    d = canonical_signing_bytes(sender="s", task_id="t", session_id="", message="hi", timestamp="2026-01-01T00:00:00+00:00")
    assert c == d


# --------------------------------------------------------------------------
# verify_envelope — real hybrid round-trip
# --------------------------------------------------------------------------


def test_sign_then_verify_roundtrip_ok():
    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    block = sign_envelope(kp, sender=DID, task_id="task-1", message="ping", timestamp=ts, session_id="s1")
    # v2 blocks carry a per-envelope nonce; the verifier reconstructs the bytes
    # with it (verify_inbound_envelope reads it from the block automatically).
    assert block["v"] == 2 and block["nonce"]
    v = verify_envelope(doc, block, sender=DID, task_id="task-1", message="ping", timestamp=ts, session_id="s1", nonce=block["nonce"])
    assert v.ok is True
    assert v.verified is True


def test_sign_then_verify_roundtrip_with_bound_fields_ok():
    """skill/verb/reply/artifacts/causation_chain are bound and round-trip."""
    from kestrel_sovereign.a2a.envelope_signing import bound_envelope_fields

    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    md = {"skill": "review", "a2a_verb": "ask", "reply_expected": True,
          "causation_chain": ["a", "b"]}
    bound = bound_envelope_fields(md, artifacts=[{"name": "x", "parts": ["p"]}])
    block = sign_envelope(kp, sender=DID, task_id="t", message="m", timestamp=ts, bound=bound)
    v = verify_envelope(doc, block, sender=DID, task_id="t", message="m", timestamp=ts, bound=bound, nonce=block["nonce"])
    assert v.ok is True and v.verified is True


def test_tampered_bound_field_fails():
    """Rewriting the causation_chain (loop-detection lineage) on an otherwise
    valid envelope must fail verification (#1721)."""
    from kestrel_sovereign.a2a.envelope_signing import bound_envelope_fields

    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    signed_bound = bound_envelope_fields({"causation_chain": ["a", "b"]})
    block = sign_envelope(kp, sender=DID, task_id="t", message="m", timestamp=ts, bound=signed_bound)
    # Attacker resets the chain to escape the depth-2 cycle guard.
    forged_bound = bound_envelope_fields({"causation_chain": []})
    v = verify_envelope(doc, block, sender=DID, task_id="t", message="m", timestamp=ts, bound=forged_bound, nonce=block["nonce"])
    assert v.ok is False


def test_tampered_message_fails():
    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    block = sign_envelope(kp, sender=DID, task_id="task-1", message="ping", timestamp=ts)
    v = verify_envelope(doc, block, sender=DID, task_id="task-1", message="PONG", timestamp=ts)
    assert v.ok is False
    assert v.verified is False


def test_stripped_pq_signature_fails_hybrid_required():
    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    block = sign_envelope(kp, sender=DID, task_id="task-1", message="ping", timestamp=ts)
    # Drop the post-quantum signature — HYBRID_REQUIRED must reject.
    block["signatures"] = [s for s in block["signatures"] if s["alg"] == "ed25519"]
    v = verify_envelope(doc, block, sender=DID, task_id="task-1", message="ping", timestamp=ts)
    assert v.ok is False


def test_stale_timestamp_fails():
    kp, doc = _keypair_and_doc()
    old = _now_iso(datetime.now(timezone.utc) - timedelta(seconds=3600))
    block = sign_envelope(kp, sender=DID, task_id="task-1", message="ping", timestamp=old)
    v = verify_envelope(doc, block, sender=DID, task_id="task-1", message="ping", timestamp=old)
    assert v.ok is False
    assert "stale" in v.reason


def test_future_timestamp_fails():
    kp, doc = _keypair_and_doc()
    future = _now_iso(datetime.now(timezone.utc) + timedelta(seconds=3600))
    block = sign_envelope(kp, sender=DID, task_id="task-1", message="ping", timestamp=future)
    v = verify_envelope(doc, block, sender=DID, task_id="task-1", message="ping", timestamp=future)
    assert v.ok is False
    assert "future" in v.reason


def test_did_document_binding_mismatch_fails():
    """A valid signature under a DID doc whose id != claimed sender is rejected
    (can't authenticate sender A with agent B's document)."""
    kp, doc = _keypair_and_doc(did="did:web:example.com:agent:other")
    ts = _now_iso()
    block = sign_envelope(kp, sender="did:web:example.com:agent:other", task_id="t", message="m", timestamp=ts)
    v = verify_envelope(doc, block, sender=DID, task_id="t", message="m", timestamp=ts)
    assert v.ok is False
    assert "does not match sender" in v.reason


def test_wrong_key_fails():
    """A signature from a different keypair fails against the victim's doc."""
    _, doc = _keypair_and_doc()
    attacker, _ = _keypair_and_doc()
    ts = _now_iso()
    block = sign_envelope(attacker, sender=DID, task_id="t", message="m", timestamp=ts)
    v = verify_envelope(doc, block, sender=DID, task_id="t", message="m", timestamp=ts)
    assert v.ok is False


# --------------------------------------------------------------------------
# verify_inbound_envelope — back-compat decision matrix
# --------------------------------------------------------------------------


def _signed_metadata(kp, *, sender=DID, task_id="t", message="m", session_id=None, ts=None):
    from kestrel_sovereign.a2a.envelope_signing import bound_envelope_fields

    ts = ts or _now_iso()
    meta = {"sender": sender, "session_id": session_id}
    # Mirror the real signer (peers.feature._maybe_sign_outbound): bind the
    # behaviour-steering fields derived from the same metadata so the inbound
    # verifier reconstructs identical bytes.
    bound = bound_envelope_fields(meta, artifacts=None)
    meta["signature"] = sign_envelope(
        kp, sender=sender, task_id=task_id, message=message, timestamp=ts,
        session_id=session_id, bound=bound,
    )
    return meta


def test_unsigned_allowed_by_default():
    v = asyncio.run(verify_inbound_envelope({"sender": "claw"}, task_id="t", message="m"))
    assert v.ok is True and v.verified is False


def test_unsigned_rejected_when_require_signed():
    v = asyncio.run(verify_inbound_envelope({"sender": "claw"}, task_id="t", message="m", require_signed=True))
    assert v.ok is False


def test_signed_and_resolvable_verifies():
    kp, doc = _keypair_and_doc()
    meta = _signed_metadata(kp)

    def resolver(did):
        return doc if did == DID else None

    v = asyncio.run(verify_inbound_envelope(meta, task_id="t", message="m", resolver=resolver))
    assert v.ok is True and v.verified is True


def test_signed_async_resolver_verifies():
    kp, doc = _keypair_and_doc()
    meta = _signed_metadata(kp)

    async def resolver(did):
        await asyncio.sleep(0)
        return doc

    v = asyncio.run(verify_inbound_envelope(meta, task_id="t", message="m", resolver=resolver))
    assert v.ok is True and v.verified is True


def test_signed_but_unresolvable_rejected_by_default():
    """A present signature whose sender can't be resolved is a FAILURE, never
    downgraded to 'allowed unsigned' (#1721) — even when require_signed=False."""
    kp, _ = _keypair_and_doc()
    meta = _signed_metadata(kp)
    v = asyncio.run(verify_inbound_envelope(meta, task_id="t", message="m", resolver=lambda did: None))
    assert v.ok is False
    assert "cannot resolve" in v.reason


def test_signed_but_unresolvable_rejected_when_require_signed():
    kp, _ = _keypair_and_doc()
    meta = _signed_metadata(kp)
    v = asyncio.run(verify_inbound_envelope(meta, task_id="t", message="m", resolver=lambda did: None, require_signed=True))
    assert v.ok is False


def test_replay_of_same_envelope_rejected():
    """A second verify of the same signed envelope (same nonce) inside the
    window is rejected — verify atomically RESERVES the nonce (#1721)."""
    from kestrel_sovereign.a2a.envelope_signing import ReplayGuard

    kp, doc = _keypair_and_doc()
    meta = _signed_metadata(kp)
    guard = ReplayGuard()
    first = asyncio.run(verify_inbound_envelope(
        meta, task_id="t", message="m", resolver=lambda did: doc, replay_guard=guard))
    assert first.ok is True and first.verified is True
    second = asyncio.run(verify_inbound_envelope(
        meta, task_id="t", message="m", resolver=lambda did: doc, replay_guard=guard))
    assert second.ok is False
    assert "repla" in second.reason.lower()


def test_rollback_after_failure_allows_retry():
    """Rolling back the reservation (downstream task creation failed) lets a
    legitimate retry of the same signed body verify again — replay protection
    must not break ordinary retries (#1721 codex r2/r3)."""
    from kestrel_sovereign.a2a.envelope_signing import ReplayGuard, rollback_envelope_nonce

    kp, doc = _keypair_and_doc()
    meta = _signed_metadata(kp)
    guard = ReplayGuard()
    first = asyncio.run(verify_inbound_envelope(
        meta, task_id="t", message="m", resolver=lambda did: doc, replay_guard=guard))
    assert first.ok is True
    rollback_envelope_nonce(first, replay_guard=guard)  # simulate create_task failure
    retry = asyncio.run(verify_inbound_envelope(
        meta, task_id="t", message="m", resolver=lambda did: doc, replay_guard=guard))
    assert retry.ok is True and retry.verified is True


def test_causation_chain_type_substitution_rejected():
    """An attacker can't erase lineage by swapping each frame dict for its string
    repr: the bound chain is structure-preserving, so the bytes (and signature)
    change (#1721 codex r3 P1)."""
    from kestrel_sovereign.a2a.envelope_signing import bound_envelope_fields

    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    frames = [{"agent": "a", "source": "s", "depth": 1}]
    signed_bound = bound_envelope_fields({"causation_chain": frames})
    block = sign_envelope(kp, sender=DID, task_id="t", message="m", timestamp=ts, bound=signed_bound)
    # Swap the frame dict for its exact string repr — a flattened projection
    # would collide; the structure-preserving one must not.
    forged_bound = bound_envelope_fields({"causation_chain": [str(frames[0])]})
    v = verify_envelope(doc, block, sender=DID, task_id="t", message="m", timestamp=ts, bound=forged_bound, nonce=block["nonce"])
    assert v.ok is False


def test_inbound_binds_metadata_skill_and_artifacts():
    """The inbound verifier binds the behaviour-steering metadata + top-level
    artifacts; tampering either after signing fails verification."""
    from kestrel_sovereign.a2a.envelope_signing import bound_envelope_fields

    kp, doc = _keypair_and_doc()
    ts = _now_iso()
    md = {"sender": DID, "skill": "review", "a2a_verb": "ask"}
    artifacts = [{"name": "report", "parts": ["original"]}]
    bound = bound_envelope_fields(md, artifacts=artifacts)
    md["signature"] = sign_envelope(kp, sender=DID, task_id="t", message="m", timestamp=ts, bound=bound)
    # Honest path verifies.
    ok = asyncio.run(verify_inbound_envelope(
        md, task_id="t", message="m", artifacts=artifacts, resolver=lambda did: doc))
    assert ok.ok is True and ok.verified is True
    # Tampered artifact text fails.
    bad = asyncio.run(verify_inbound_envelope(
        md, task_id="t", message="m",
        artifacts=[{"name": "report", "parts": ["INJECTED"]}],
        resolver=lambda did: doc, replay_guard=None))
    assert bad.ok is False


def test_signed_but_bad_signature_always_rejected_even_without_require():
    """A present-but-invalid signature is an attack signal — never downgraded
    to 'unsigned', even when require_signed is False."""
    kp, doc = _keypair_and_doc()
    meta = _signed_metadata(kp, message="m")
    # Verify against a DIFFERENT message than what was signed.
    v = asyncio.run(verify_inbound_envelope(meta, task_id="t", message="TAMPERED", resolver=lambda did: doc))
    assert v.ok is False


def test_signature_present_but_no_sender_rejected():
    kp, _ = _keypair_and_doc()
    block = sign_envelope(kp, sender=DID, task_id="t", message="m", timestamp=_now_iso())
    v = asyncio.run(verify_inbound_envelope({"signature": block}, task_id="t", message="m"))
    assert v.ok is False


@pytest.mark.parametrize("bad", [{}, [], "", {"alg": "hybrid-v2"}])
def test_malformed_signature_block_rejected_without_resolver(bad):
    """A present-but-malformed block is rejected up front — never downgraded to
    'unsigned' even when no resolver is wired and require_signed is False."""
    v = asyncio.run(verify_inbound_envelope(
        {"sender": DID, "signature": bad}, task_id="t", message="m", resolver=None
    ))
    assert v.ok is False
    assert "malformed" in v.reason
