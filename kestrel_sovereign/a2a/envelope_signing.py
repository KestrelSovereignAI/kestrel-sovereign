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
import secrets
import threading
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
# v2 (#1721): binds the behaviour-steering fields (skill, a2a_verb,
# reply_expected, artifacts, causation_chain) and a per-envelope nonce in
# addition to the v1 core (sender, task_id, session_id, message, timestamp).
ENVELOPE_SIG_VERSION = 2
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
    # The verified sender DID and per-envelope nonce, surfaced on the verdict for
    # observability/audit. Empty for unsigned/failed verdicts.
    sender: str = ""
    nonce: str = ""


def canonical_message(part_texts: "list[str]") -> str:
    """Structure-preserving canonical form of a message's text parts.

    Signed/verified as the ``message`` field. A JSON array (not a ``\\n``-join)
    so multipart structure can't be forged: ``["a\\nb"]`` and ``["a", "b"]``
    produce distinct bytes even though a flattened join would collide.
    """
    return json.dumps(list(part_texts), ensure_ascii=False, separators=(",", ":"))


def _canonical_chain(chain: Any) -> Any:
    """Normalise a causation chain to a STRUCTURE-PRESERVING form for binding.

    The chain lives in ``metadata["causation_chain"]`` as a serialized list of
    frame dicts; both signer and verifier project it identically so the
    loop-detection lineage can't be rewritten on an otherwise-valid envelope.

    Structure must be preserved (NOT ``str()``-flattened): a flattened
    projection lets an attacker swap each frame dict for its exact string repr —
    the bound bytes would match and the signature verify, but ``_deserialize_chain``
    later drops the non-dict entries, erasing the lineage. A JSON round-trip
    keeps dict/list shape distinct from a string while staying serialisable.
    """
    if not isinstance(chain, (list, tuple)):
        return []
    try:
        return json.loads(json.dumps(list(chain), default=str))
    except (TypeError, ValueError):
        return [str(x) for x in chain]


def _canonical_artifacts(artifacts: Any) -> Any:
    """Normalise artifacts to a JSON-stable structure for binding.

    Accepts a list of mappings (or objects exposing ``model_dump``) and returns
    plain data that ``json.dumps(sort_keys=True)`` renders deterministically, so
    an attacker can't append/alter artifacts on a signed envelope. Non-list
    inputs normalise to ``[]`` so a missing optional field is stable.
    """
    if not isinstance(artifacts, (list, tuple)):
        return []
    out: "list[Any]" = []
    for a in artifacts:
        if hasattr(a, "model_dump"):
            try:
                out.append(a.model_dump(mode="json"))
                continue
            except Exception:  # noqa: BLE001 - fall through to best-effort
                pass
        out.append(a)
    return out


def bound_envelope_fields(
    metadata: Optional[Mapping[str, Any]],
    *,
    artifacts: Any = None,
) -> "dict[str, Any]":
    """The behaviour-steering fields bound into a v2 signature.

    Both signer and verifier call this on their own view of the request so the
    bytes are byte-identical by construction (no per-call drift). ``skill`` is
    read under either ``skill`` or ``skill_id`` (both spellings appear on the
    wire). ``artifacts`` is passed explicitly because it is a top-level
    ``TaskSendParams`` field, not part of ``metadata``.
    """
    md = metadata or {}
    return {
        "skill": str(md.get("skill") or md.get("skill_id") or ""),
        "a2a_verb": str(md.get("a2a_verb") or ""),
        "reply_expected": bool(md.get("reply_expected", False)),
        "causation_chain": _canonical_chain(md.get("causation_chain")),
        "artifacts": _canonical_artifacts(artifacts),
    }


def canonical_signing_bytes(
    *,
    sender: str,
    task_id: str,
    session_id: Optional[str],
    message: str,
    timestamp: str,
    nonce: str = "",
    bound: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """Deterministic byte view of the signed envelope fields.

    Both signer and verifier MUST produce identical bytes, so this is a
    sorted-key, compact-separator JSON over a fixed field set plus a version
    tag. ``None`` session ids are normalised to ``""`` so a missing optional
    field doesn't change the bytes between signer and verifier. ``bound`` is the
    :func:`bound_envelope_fields` projection of the behaviour-steering fields;
    ``nonce`` is the per-envelope replay nonce. Both are signed so neither can
    be altered on an otherwise-valid envelope.
    """
    payload = {
        "_v": ENVELOPE_SIG_VERSION,
        "sender": sender,
        "task_id": task_id,
        "session_id": session_id or "",
        "message": message,
        "timestamp": timestamp,
        "nonce": nonce or "",
        "bound": dict(bound or {}),
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def kids_from_verification_methods(
    verification_methods: "list[Mapping[str, Any]]",
) -> "tuple[str, str]":
    """Derive ``(classical_kid, pq_kid)`` from a DID document's VM list.

    The verifier matches a signature entry's ``kid`` to a VM ``id`` fragment, so
    the signer must use kids that match its *published* verification methods.
    VMs are classical-first, PQ-second (``build_verification_methods`` ordering),
    so we take the fragment (after ``#``) of VM[0] and VM[1]. Falls back to the
    ``sign_hybrid`` defaults (``key-1`` / ``key-2``) when a VM is missing — those
    are exactly the kids the inception/rotation ceremonies assign.
    """
    def _frag(vm: Any, default: str) -> str:
        if isinstance(vm, Mapping):
            vm_id = vm.get("id") or ""
            if "#" in vm_id:
                return vm_id.rsplit("#", 1)[-1]
        return default

    vms = list(verification_methods or [])
    classical = _frag(vms[0] if len(vms) > 0 else None, "key-1")
    pq = _frag(vms[1] if len(vms) > 1 else None, "key-2")
    return classical, pq


def sign_envelope(
    keypair: HybridKeypair,
    *,
    sender: str,
    task_id: str,
    message: str,
    timestamp: str,
    session_id: Optional[str] = None,
    bound: Optional[Mapping[str, Any]] = None,
    nonce: Optional[str] = None,
    classical_kid: str = "key-1",
    pq_kid: str = "key-2",
) -> dict:
    """Produce the signature block to attach to an outbound envelope.

    The returned dict drops into ``metadata["signature"]``::

        {"alg": "hybrid-v2", "v": 2, "timestamp": "<iso>", "nonce": "<hex>",
         "signatures": [{"alg": "ed25519", "kid": "key-1", "sig": "<hex>"},
                        {"alg": "ml-dsa-65", "kid": "key-2", "sig": "<hex>"}]}

    ``sender`` should be the signer's DID (matching the DID document the
    receiver will resolve). ``timestamp`` is an ISO-8601 UTC string the
    receiver checks for freshness. ``bound`` is the
    :func:`bound_envelope_fields` projection of the behaviour-steering fields
    (skill/verb/reply/artifacts/causation_chain); a fresh ``nonce`` is generated
    when not supplied so the receiver can reject verbatim replays inside the
    freshness window. ``classical_kid`` / ``pq_kid`` must match the signer's
    published verification-method ids — derive them with
    :func:`kids_from_verification_methods` when signing as a real identity.
    """
    nonce = nonce or secrets.token_hex(16)
    data = canonical_signing_bytes(
        sender=sender,
        task_id=task_id,
        session_id=session_id,
        message=message,
        timestamp=timestamp,
        nonce=nonce,
        bound=bound,
    )
    return {
        "alg": ENVELOPE_SIG_ALG,
        "v": ENVELOPE_SIG_VERSION,
        # Carried in the block so the receiver can reconstruct the canonical
        # bytes. Both are part of the signed payload, so tampering either
        # changes the bytes and fails verification (the freshness check + nonce
        # cache together bound replay).
        "timestamp": timestamp,
        "nonce": nonce,
        "signatures": sign_hybrid(data, keypair, classical_kid=classical_kid, pq_kid=pq_kid),
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
    bound: Optional[Mapping[str, Any]] = None,
    nonce: str = "",
    policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: Optional[datetime] = None,
) -> EnvelopeVerification:
    """Verify a signature block against the sender's resolved DID document.

    Checks, in order: timestamp freshness (replay window), DID-document binding
    (``did_document["id"]`` must equal the claimed ``sender`` — a doc for a
    different DID can't authenticate this sender), then the hybrid signature
    over the canonical bytes (which now include ``nonce`` and the ``bound``
    behaviour-steering fields) against the document's ``verificationMethod``
    entries under ``policy``.
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
        nonce=nonce,
        bound=bound,
    )
    result = verify_hybrid(data, signatures, verification_methods, policy=policy)
    if not result.ok:
        return EnvelopeVerification(ok=False, reason=f"signature check failed: {result.reason}")
    return EnvelopeVerification(ok=True, reason=result.reason, verified=True)


class ReplayGuard:
    """Bounded, thread-safe nonce reservation cache for replay rejection.

    A signed envelope carries a per-envelope ``nonce``; the freshness window
    (±``ttl_seconds``) bounds how long a captured envelope is replayable, and
    this guard rejects a re-submission of the same ``(sender, nonce)`` inside
    that window. Entries are pruned past ``ttl_seconds`` so the cache stays small.

    Consume-on-reserve semantics: :meth:`reserve` atomically checks-and-records
    under one lock, so two concurrent submissions of the same body can't both
    pass (the second loses the race) and a later replay of the same body is
    rejected. A reserved nonce is *spent* — there is deliberately no rollback,
    because downstream task creation is NOT idempotent (it appends a session
    event), so releasing the nonce on a partial-create failure could double-
    process (codex r5). A client retrying after a server error must send a
    freshly-signed envelope — Kestrel peers re-sign every send with a new nonce +
    timestamp — so this trades exact-body retryability for leak-free protection
    without requiring idempotent task creation.

    Scope: this is a PER-PROCESS guard. It fully protects a single-process
    deployment; in a multi-worker / multi-instance deployment a captured envelope
    replayed to a *different* worker is not caught here. Shared cross-worker
    replay state (a DB/Redis-backed nonce store) is tracked as follow-up
    hardening (see epic #1720).
    """

    def __init__(self, ttl_seconds: int = DEFAULT_MAX_AGE_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._seen: "dict[str, float]" = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(sender: str, nonce: str) -> str:
        return f"{sender}\x00{nonce}"

    def reserve(self, sender: str, nonce: str, *, now_ts: float) -> bool:
        """Atomically reserve a ``(sender, nonce)``. Return True if fresh (now
        reserved), False if already reserved/consumed in-window (a replay).

        An empty nonce is always fresh (replay binding is best-effort; the
        freshness window still applies). The check and the record happen under a
        single lock so concurrent duplicates can't both succeed."""
        if not nonce:
            return True
        key = self._key(sender, nonce)
        with self._lock:
            cutoff = now_ts - self._ttl
            # Opportunistic prune of expired entries.
            if len(self._seen) > 1024:
                self._seen = {k: t for k, t in self._seen.items() if t >= cutoff}
            seen_at = self._seen.get(key)
            if seen_at is not None and seen_at >= cutoff:
                return False
            self._seen[key] = now_ts
            return True


# Process-wide default guard used by the inbound endpoint when no guard is
# injected. Per-process is sufficient: replay protection only needs to span the
# freshness window on the host that actually receives the envelope.
#
# TTL = 2× the freshness window, NOT 1×: ``_timestamp_fresh`` accepts a timestamp
# up to ``max_age_seconds`` in the FUTURE (clock-skew tolerance) and then for
# ``max_age_seconds`` after it, so a single envelope can be validly replayable
# for up to 2× the window measured from first receipt. A 1× TTL would prune the
# reservation while the signature is still fresh, reopening a replay gap (codex
# r4). The guard must outlive the full validity window.
REPLAY_GUARD_TTL_SECONDS = 2 * DEFAULT_MAX_AGE_SECONDS
_DEFAULT_REPLAY_GUARD = ReplayGuard(ttl_seconds=REPLAY_GUARD_TTL_SECONDS)


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
    artifacts: Any = None,
    resolver: Optional[Resolver] = None,
    require_signed: bool = False,
    policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    replay_guard: Optional[ReplayGuard] = _DEFAULT_REPLAY_GUARD,
    now: Optional[datetime] = None,
) -> EnvelopeVerification:
    """Decide whether to accept an inbound A2A envelope based on its signature.

    ``session_id`` is the authoritative top-level value the task is created
    under (NOT read from caller-controlled metadata) so a signature can't bind
    a different session than the task runs in. ``artifacts`` is the authoritative
    top-level ``TaskSendParams.artifacts`` list, bound into the signature so it
    can't be altered post-signing.

    Decision matrix:

    * no ``signature`` block (key absent or null):
        - ``require_signed`` → reject;
        - else → **allow unsigned** (``verified=False``) — the same-host
          API-key boundary still applies.
    * ``signature`` present but sender DID unresolvable: **reject**. A present
      signature is a verification claim; if it can't be checked it's a failure,
      never silently downgraded to "allowed unsigned" (#1721). The only
      escape hatch is genuinely *unsigned* traffic above.
    * ``signature`` present and sender resolvable: cryptographically verify
      (including the freshness window and replay nonce); any failure is rejected.

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
        # A present signature that cannot be resolved is a FAILURE, not benign
        # absence — never downgrade to "allowed unsigned" (#1721). An attacker
        # who attaches a garbage signature for an unknown DID must not be
        # treated more leniently than one who sends a verifiable bad signature.
        logger.warning(
            "A2A: rejecting signed envelope from %r: sender DID is unresolvable.",
            sender,
        )
        return EnvelopeVerification(
            ok=False, reason=f"cannot resolve sender DID {sender!r}"
        )

    timestamp = signature_block.get("timestamp") or ""
    nonce = str(signature_block.get("nonce") or "")
    bound = bound_envelope_fields(metadata, artifacts=artifacts)
    verdict = verify_envelope(
        did_document,
        signature_block,
        sender=sender,
        task_id=task_id,
        message=message,
        timestamp=timestamp,
        session_id=session_id,
        bound=bound,
        nonce=nonce,
        policy=policy,
        max_age_seconds=max_age_seconds,
        now=now,
    )
    if not verdict.ok:
        logger.warning("A2A: rejecting signed envelope from %r: %s", sender, verdict.reason)
        return verdict

    # Signature is valid; ATOMICALLY reserve the (sender, nonce) to reject both
    # verbatim replays and concurrent duplicate submissions inside the freshness
    # window. The reservation is the consume — deliberately no rollback (see
    # ReplayGuard docstring): a retry after a downstream failure must be a
    # freshly-signed envelope, which Kestrel peers produce automatically.
    if replay_guard is not None and nonce:
        now_dt = now or datetime.now(timezone.utc)
        if not replay_guard.reserve(sender, nonce, now_ts=now_dt.timestamp()):
            logger.warning("A2A: rejecting replayed envelope from %r (nonce reuse).", sender)
            return EnvelopeVerification(
                ok=False, reason="replayed envelope (nonce already seen in window)"
            )
    # Surface the verified sender+nonce on the verdict for observability/audit.
    return EnvelopeVerification(
        ok=verdict.ok, reason=verdict.reason, verified=verdict.verified,
        sender=sender, nonce=nonce,
    )
