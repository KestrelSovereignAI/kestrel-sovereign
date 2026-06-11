"""Cryptographic sender authentication for A2A task envelopes (#1673).

Inbound ``POST /tasks/send`` accepts ``metadata["sender"]`` as an **unverified
string claim**. The v1 trust model is the same-host shared-API-key boundary
(the endpoint sits behind the host API-key middleware), so today that's
*unverified attribution within a trusted boundary*. This module adds the
cryptographic layer the follow-up epic promised: a sender signs a canonical
view of the envelope with its hybrid (Ed25519 + ML-DSA-65) identity key, and a
receiver verifies that signature against the sender's DID document.

**Resolution topology (the #1673 design).** This module is deliberately
*resolver-agnostic*: ``verify_envelope`` takes an already-resolved DID
document, and ``verify_inbound_envelope`` takes a ``resolver`` callable. That
keeps the policy here — *local same-host resolution by default, federated
``did:web`` optional, never required*. Who resolves (a host registry of peer
DID docs) and who signs on send (the agent's runtime keypair) are separate
wiring concerns; this module is the shared, fully-testable contract both sides
depend on.

**Back-compat.** ``verify_inbound_envelope`` allows an *unsigned* envelope
through by default (preserving the same-host boundary) and only rejects when
a signature is present and fails, or when ``require_signed=True``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

from kestrel_sovereign.identity.hybrid_keypair import (
    HybridKeypair,
    sign_hybrid,
    verify_hybrid,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy

logger = logging.getLogger(__name__)

# Bumped if the canonical-bytes layout changes; both sides must agree.
ENVELOPE_SIG_VERSION = 1
# Algorithm tag for the signature block (hybrid v2 signatures array).
ENVELOPE_SIG_ALG = "hybrid-v2"
# Default replay window: a signed envelope older than this is rejected.
DEFAULT_MAX_AGE_SECONDS = 300

# A resolver maps a sender DID -> its DID document (or None if unknown). It may
# be sync or async; ``verify_inbound_envelope`` awaits awaitables.
Resolver = Callable[[str], Union[Optional[Mapping[str, Any]], Awaitable[Optional[Mapping[str, Any]]]]]


@dataclass(frozen=True)
class EnvelopeVerification:
    """Verdict from verifying a signed envelope.

    ``ok`` is the boolean verdict; ``reason`` is human-readable; ``verified``
    distinguishes "cryptographically verified" from "allowed unsigned" so a
    caller can apply a stricter governance tier to verified peers.
    """

    ok: bool
    reason: str
    verified: bool = False


def canonical_message(part_texts: "list[str]") -> str:
    """Structure-preserving canonical form of a message's text parts.

    Signed/verified as the ``message`` field. A JSON array (not a ``\\n``-join)
    so multipart structure can't be forged: ``["a\\nb"]`` and ``["a", "b"]``
    produce distinct bytes even though a flattened join would collide.
    """
    return json.dumps(list(part_texts), ensure_ascii=False, separators=(",", ":"))


def canonical_signing_bytes(
    *,
    sender: str,
    task_id: str,
    session_id: Optional[str],
    message: str,
    timestamp: str,
) -> bytes:
    """Deterministic byte view of the signed envelope fields.

    Both signer and verifier MUST produce identical bytes, so this is a
    sorted-key, compact-separator JSON over a fixed field set plus a version
    tag. ``None`` session ids are normalised to ``""`` so a missing optional
    field doesn't change the bytes between signer and verifier.
    """
    payload = {
        "_v": ENVELOPE_SIG_VERSION,
        "sender": sender,
        "task_id": task_id,
        "session_id": session_id or "",
        "message": message,
        "timestamp": timestamp,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sign_envelope(
    keypair: HybridKeypair,
    *,
    sender: str,
    task_id: str,
    message: str,
    timestamp: str,
    session_id: Optional[str] = None,
) -> dict:
    """Produce the signature block to attach to an outbound envelope.

    The returned dict drops into ``metadata["signature"]``::

        {"alg": "hybrid-v2", "v": 1,
         "signatures": [{"alg": "ed25519", "kid": "key-1", "sig": "<hex>"},
                        {"alg": "ml-dsa-65", "kid": "key-2", "sig": "<hex>"}]}

    ``sender`` should be the signer's DID (matching the DID document the
    receiver will resolve). ``timestamp`` is an ISO-8601 UTC string the
    receiver checks for freshness.
    """
    data = canonical_signing_bytes(
        sender=sender,
        task_id=task_id,
        session_id=session_id,
        message=message,
        timestamp=timestamp,
    )
    return {
        "alg": ENVELOPE_SIG_ALG,
        "v": ENVELOPE_SIG_VERSION,
        # Carried in the block so the receiver can reconstruct the canonical
        # bytes. It is part of the signed payload, so tampering it changes the
        # bytes and fails verification (and the freshness check bounds replay).
        "timestamp": timestamp,
        "signatures": sign_hybrid(data, keypair),
    }


def _timestamp_fresh(timestamp: str, *, max_age_seconds: int, now: datetime) -> Optional[str]:
    """Return an error string if the timestamp is missing/old/future, else None."""
    if not timestamp:
        return "missing timestamp"
    try:
        ts = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return f"unparseable timestamp {timestamp!r}"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (now - ts).total_seconds()
    if age > max_age_seconds:
        return f"stale envelope ({int(age)}s old > {max_age_seconds}s window)"
    # Allow modest clock skew into the future.
    if age < -max_age_seconds:
        return f"envelope timestamp is {int(-age)}s in the future"
    return None


def verify_envelope(
    did_document: Mapping[str, Any],
    signature_block: Mapping[str, Any],
    *,
    sender: str,
    task_id: str,
    message: str,
    timestamp: str,
    session_id: Optional[str] = None,
    policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: Optional[datetime] = None,
) -> EnvelopeVerification:
    """Verify a signature block against the sender's resolved DID document.

    Checks, in order: timestamp freshness (replay window), DID-document binding
    (``did_document["id"]`` must equal the claimed ``sender`` — a doc for a
    different DID can't authenticate this sender), then the hybrid signature
    against the document's ``verificationMethod`` entries under ``policy``.
    """
    now = now or datetime.now(timezone.utc)

    stale = _timestamp_fresh(timestamp, max_age_seconds=max_age_seconds, now=now)
    if stale:
        return EnvelopeVerification(ok=False, reason=stale)

    doc_id = did_document.get("id")
    if doc_id != sender:
        return EnvelopeVerification(
            ok=False,
            reason=f"DID document id {doc_id!r} does not match sender {sender!r}",
        )

    signatures = signature_block.get("signatures") if isinstance(signature_block, Mapping) else None
    if not signatures:
        return EnvelopeVerification(ok=False, reason="signature block has no signatures")

    verification_methods = did_document.get("verificationMethod") or []
    if not verification_methods:
        return EnvelopeVerification(
            ok=False, reason="DID document has no verificationMethod entries"
        )

    data = canonical_signing_bytes(
        sender=sender,
        task_id=task_id,
        session_id=session_id,
        message=message,
        timestamp=timestamp,
    )
    result = verify_hybrid(data, signatures, verification_methods, policy=policy)
    if not result.ok:
        return EnvelopeVerification(ok=False, reason=f"signature check failed: {result.reason}")
    return EnvelopeVerification(ok=True, reason=result.reason, verified=True)


async def _resolve(resolver: Optional[Resolver], did: str) -> Optional[Mapping[str, Any]]:
    if resolver is None:
        return None
    out = resolver(did)
    if hasattr(out, "__await__"):
        out = await out  # type: ignore[assignment]
    return out  # type: ignore[return-value]


async def verify_inbound_envelope(
    metadata: Mapping[str, Any],
    *,
    task_id: str,
    message: str,
    session_id: Optional[str] = None,
    resolver: Optional[Resolver] = None,
    require_signed: bool = False,
    policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: Optional[datetime] = None,
) -> EnvelopeVerification:
    """Decide whether to accept an inbound A2A envelope based on its signature.

    ``session_id`` is the authoritative top-level value the task is created
    under (NOT read from caller-controlled metadata) so a signature can't bind
    a different session than the task runs in.

    Decision matrix (back-compat by default):

    * no ``signature`` block (key absent or null):
        - ``require_signed`` → reject;
        - else → **allow unsigned** (``verified=False``) — the same-host
          API-key boundary still applies.
    * ``signature`` present but sender DID unresolvable:
        - ``require_signed`` → reject;
        - else → allow unsigned, logging that the claim couldn't be verified.
    * ``signature`` present and sender resolvable: cryptographically verify;
      a failure is **always** rejected (a present-but-bad signature is an
      attack signal, never downgraded to "unsigned").

    A present-but-structurally-malformed block (``{}``, ``[]``, ``""``, or a
    mapping without ``signatures``) is rejected up front, regardless of
    resolver — it is a malformed request, never treated as "no signature".
    """
    # Presence is key-existence with a non-null value: an empty/malformed block
    # ({}, [], "") is PRESENT (and will fail verification), not "unsigned".
    signature_block = metadata.get("signature")
    sender = str(metadata.get("sender") or "")

    if signature_block is None:
        if require_signed:
            return EnvelopeVerification(
                ok=False, reason="unsigned envelope rejected (require_signed=True)"
            )
        return EnvelopeVerification(ok=True, reason="unsigned envelope allowed (same-host boundary)")

    # A present but structurally-malformed block ({}, [], "", non-mapping, or a
    # mapping with no signatures) is rejected REGARDLESS of resolver — it is a
    # malformed request, not a verifiable claim, so it must never be downgraded
    # to "unsigned" even when no resolver is wired.
    if not isinstance(signature_block, Mapping) or not signature_block.get("signatures"):
        return EnvelopeVerification(ok=False, reason="malformed signature block")

    if not sender:
        return EnvelopeVerification(ok=False, reason="signature present but no sender DID claimed")

    did_document = await _resolve(resolver, sender)
    if did_document is None:
        if require_signed:
            return EnvelopeVerification(
                ok=False, reason=f"cannot resolve sender DID {sender!r} (require_signed=True)"
            )
        logger.warning(
            "A2A: signed envelope from %r but DID is unresolvable; allowing as unsigned "
            "(same-host boundary). Provide a DID resolver to enforce verification.",
            sender,
        )
        return EnvelopeVerification(
            ok=True, reason=f"sender DID {sender!r} unresolvable; allowed as unsigned"
        )

    timestamp = signature_block.get("timestamp") if isinstance(signature_block, Mapping) else ""
    verdict = verify_envelope(
        did_document,
        signature_block if isinstance(signature_block, Mapping) else {},
        sender=sender,
        task_id=task_id,
        message=message,
        timestamp=timestamp or "",
        session_id=session_id,
        policy=policy,
        max_age_seconds=max_age_seconds,
        now=now,
    )
    if not verdict.ok:
        logger.warning("A2A: rejecting signed envelope from %r: %s", sender, verdict.reason)
    return verdict
