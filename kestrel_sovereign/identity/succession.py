"""
Succession statements — Wave 3 sub-PR 2 of Quantum Hardening (#921, #918).

A **succession statement** is the cryptographic bridge that lets a
Kestrel agent rotate from one identity to another while preserving an
auditable chain of authority. Concretely it attests:

- Predecessor DID + verification methods at the time of signing
- Successor DID + verification methods (a hybrid did:web identity)
- ``effective_from`` — the temporal cutoff after which only the
  successor's keys are authoritative
- Predecessor signatures (proof of authorization)
- Successor signatures (proof of acceptance)
- Optional archival countersignature (SLH-DSA-SHA2-128s — hash-based,
  long-horizon durable)

Why succession statements exist
-------------------------------

Wave 2 (#917) shipped the hybrid did:web identity path but left
existing legacy agents (Kestrel #1, Emma, Meridian, Frinz tenants,
all carrying ``did:pkh:eip155:1:0x…`` ECDSA-only identities) on the
old shape. Wave 3's job is to migrate those agents to hybrid without
losing continuity. Each migration produces one succession statement,
signed by the legacy key and accepted by the new hybrid keys.

Why the temporal-cutoff matters
-------------------------------

If we accepted a classical ECDSA signature from the predecessor key
*after* the succession's ``effective_from``, a future Shor-equipped
adversary could:

1. Recover the ECDSA private key from the public key (Shor on EC).
2. Forge a back-dated signature from the predecessor authorizing a
   *different* successor.
3. Present that forgery as the canonical succession.

The mitigation is the ``post_cutoff_classical_allowed=False`` hook
in :mod:`kestrel_sovereign.security.verify_policy`: any artifact
dated after the cutoff MUST carry at least one post-quantum signature.
Wave 3 sub-PR 3 (chain walker) is what calls into this hook on
behalf of every consumer.

Why both parties must sign
--------------------------

Authorization (predecessor) without acceptance (successor) lets a
malicious predecessor "donate" their identity to an attacker-
controlled key. Acceptance without authorization lets a successor
claim provenance they don't have. Requiring both produces a
two-party-attested artifact: only when both signatures verify is the
succession trusted.

The optional **archival signature** is a third, independent SLH-DSA
signature over the same payload. Its purpose is conservative-tier
durability: even if both Wave 2 hybrid suites (Ed25519 + ML-DSA-65)
fall to a future cryptanalytic surprise, the SLH-DSA hash-based sig
remains the most defensible long-horizon signature we can ship today.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.security.crypto_suite import (
    ALG_SLH_DSA_SHA2_128S,
    CryptoSuite,
    CryptoSuiteError,
    Keypair,
    get_suite,
)
from kestrel_sovereign.security.multikey import multibase_to_public_key
from kestrel_sovereign.security.verify_policy import (
    PolicyResult,
    VerifyPolicy,
    evaluate_signatures,
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuccessionStatement:
    """An immutable succession-statement record.

    Each field is committed in the canonical signable payload (see
    :func:`signable_payload`). Signatures and identifiers are derived
    fields populated by :func:`sign_succession` / :func:`finalize`.

    Field semantics
    ---------------

    - ``predecessor_did`` / ``successor_did``: DID URIs (any method —
      ``did:pkh``, ``did:web``, ``did:key`` — but Wave 3 ceremonies
      will be ``did:pkh`` → ``did:web``).
    - ``effective_from``: ISO 8601 UTC timestamp. Verifiers compare
      this to artifact timestamps to decide whether the predecessor's
      classical-only signatures are still trustworthy.
    - ``reason``: free-text human-facing description of the rotation
      (e.g. "PQ-hardening migration per Quantum Hardening epic
      #921 Wave 3"). Committed to the signable payload so it cannot
      be retconned.
    - ``predecessor_verification_methods`` / ``successor_verification_methods``:
      W3C Multikey verification-method dicts (same shape as
      :func:`identity.did_web.build_verification_methods` produces).
      Embedded so the verifier never needs an out-of-band key
      lookup at chain-walk time.
    - ``predecessor_signatures`` / ``successor_signatures``:
      v2 ``signatures`` array entries ``{alg, kid, sig}``. Predecessor
      may be legacy (one ECDSA entry); successor is hybrid (Ed25519 +
      ML-DSA-65).
    - ``archival_signature``: optional single SLH-DSA-SHA2-128s entry.
      Provides conservative-tier long-horizon durability over the
      same signable payload. If present, must verify under the
      ``archival_verification_method``.
    - ``archival_verification_method``: optional Multikey VM for the
      SLH-DSA key that produced ``archival_signature``. Distinct kid
      so it doesn't collide with the predecessor or successor kids.
    - ``statement_id``: SHA-256 hex over ``signable_payload(self)``.
      Stable content-addressed identifier — chain walkers and audit
      logs reference this. Populated by :func:`finalize`.
    - ``created_at``: ISO 8601 UTC timestamp recording when the
      statement object was minted (NOT signed); informational only,
      excluded from the signable payload so reordering doesn't
      invalidate signatures.
    """

    predecessor_did: str
    successor_did: str
    effective_from: str
    reason: str

    predecessor_verification_methods: List[dict] = field(default_factory=list)
    successor_verification_methods: List[dict] = field(default_factory=list)

    predecessor_signatures: List[dict] = field(default_factory=list)
    successor_signatures: List[dict] = field(default_factory=list)

    archival_signature: Optional[dict] = None
    archival_verification_method: Optional[dict] = None

    statement_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """Plain-dict representation for JSON archival.

        All list and dict fields are deep-copied; the result is safe
        to mutate without affecting the underlying frozen dataclass.
        """
        return {
            "predecessor_did": self.predecessor_did,
            "successor_did": self.successor_did,
            "effective_from": self.effective_from,
            "reason": self.reason,
            "predecessor_verification_methods": [
                dict(m) for m in self.predecessor_verification_methods
            ],
            "successor_verification_methods": [
                dict(m) for m in self.successor_verification_methods
            ],
            "predecessor_signatures": [
                dict(s) for s in self.predecessor_signatures
            ],
            "successor_signatures": [
                dict(s) for s in self.successor_signatures
            ],
            "archival_signature": (
                dict(self.archival_signature) if self.archival_signature else None
            ),
            "archival_verification_method": (
                dict(self.archival_verification_method)
                if self.archival_verification_method else None
            ),
            "statement_id": self.statement_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuccessionStatement":
        return cls(
            predecessor_did=data["predecessor_did"],
            successor_did=data["successor_did"],
            effective_from=data["effective_from"],
            reason=data.get("reason", ""),
            predecessor_verification_methods=list(
                data.get("predecessor_verification_methods") or []
            ),
            successor_verification_methods=list(
                data.get("successor_verification_methods") or []
            ),
            predecessor_signatures=list(data.get("predecessor_signatures") or []),
            successor_signatures=list(data.get("successor_signatures") or []),
            archival_signature=(data.get("archival_signature") or None),
            archival_verification_method=(
                data.get("archival_verification_method") or None
            ),
            statement_id=data.get("statement_id", ""),
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Canonical signable payload
# ---------------------------------------------------------------------------
#
# Every signature is computed over EXACTLY this byte sequence. Excludes:
# - All signature fields (predecessor_signatures, successor_signatures,
#   archival_signature) — obviously, you cannot sign your own signature.
# - statement_id — derived from this payload, would be circular.
# - created_at — informational only; excluded so timestamp drift between
#   signers doesn't invalidate signatures (the binding timestamp is
#   ``effective_from``, which IS in the payload).
#
# The verification methods ARE included so each party commits to the
# exact set of keys authorized by the succession.

_SIGNED_FIELDS = (
    "predecessor_did",
    "successor_did",
    "effective_from",
    "reason",
    "predecessor_verification_methods",
    "successor_verification_methods",
)


def signable_payload(statement: SuccessionStatement) -> bytes:
    """Canonical UTF-8 bytes for signing/verification.

    Sorted-key compact JSON ensures byte-stability across signers,
    archivers, and verifiers regardless of their runtime's dict-iter
    order. The serialization is deterministic so verifiers re-derive
    the same bytes from a `to_dict()` round-trip.
    """
    payload = {
        field: getattr(statement, field)
        for field in _SIGNED_FIELDS
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def compute_statement_id(statement: SuccessionStatement) -> str:
    """SHA-256 hex of :func:`signable_payload`.

    Acts as a stable content-addressed ID: any change to a signed
    field changes the id. Audit logs and chain-walker indexes reference
    statements by this id so reorderings or partial copies can't slip
    a different statement past consumers.
    """
    return hashlib.sha256(signable_payload(statement)).hexdigest()


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def _hex_signature(suite: CryptoSuite, data: bytes, private_key: Any) -> str:
    return suite.sign(data, private_key).hex()


def sign_predecessor(
    statement: SuccessionStatement,
    predecessor_keypairs: Iterable[Tuple[Keypair, str]],
) -> SuccessionStatement:
    """Apply predecessor signatures.

    Each ``(keypair, kid)`` pair signs :func:`signable_payload` and is
    added to ``predecessor_signatures``. ``kid`` MUST match the fragment
    of one of ``predecessor_verification_methods[].id``.

    A legacy agent will pass exactly one ``(secp256k1_keypair, kid)``
    pair (their only key). A hybrid agent rotating to a new hybrid
    identity passes two pairs.
    """
    payload = signable_payload(statement)
    sigs = list(statement.predecessor_signatures)
    for kp, kid in predecessor_keypairs:
        suite = get_suite(kp.suite_id)
        sigs.append({
            "alg": suite.alg_id,
            "kid": kid,
            "sig": _hex_signature(suite, payload, kp.private_key),
        })
    return replace(statement, predecessor_signatures=sigs)


def sign_successor(
    statement: SuccessionStatement,
    successor_keypairs: Iterable[Tuple[Keypair, str]],
) -> SuccessionStatement:
    """Apply successor signatures (hybrid: Ed25519 + ML-DSA-65)."""
    payload = signable_payload(statement)
    sigs = list(statement.successor_signatures)
    for kp, kid in successor_keypairs:
        suite = get_suite(kp.suite_id)
        sigs.append({
            "alg": suite.alg_id,
            "kid": kid,
            "sig": _hex_signature(suite, payload, kp.private_key),
        })
    return replace(statement, successor_signatures=sigs)


def archival_countersign(
    statement: SuccessionStatement,
    slh_dsa_keypair: Keypair,
    *,
    kid: str = "archival",
    verification_method: Optional[dict] = None,
) -> SuccessionStatement:
    """Apply the optional SLH-DSA-SHA2-128s archival countersignature.

    ``slh_dsa_keypair`` must be from :class:`SLHDSASHA2128sSuite`.
    The countersignature is over the same canonical payload as the
    predecessor / successor signatures. ``verification_method`` is
    embedded into the statement so a long-horizon verifier can
    re-derive the public key without an external lookup; if not
    provided, callers MUST attach it before archival.

    Why a separate slot (not just another entry in successor_signatures):
    SLH-DSA's role here is **conservative-tier durability**, not
    successor authorization. Putting it in the successor array would
    let it satisfy HYBRID_REQUIRED on its own (PQ + classical from
    the same actor) and obscure the policy intent. A distinct slot
    means policy code can require it as a separate dimension.
    """
    if slh_dsa_keypair.suite_id != ALG_SLH_DSA_SHA2_128S:
        raise CryptoSuiteError(
            f"archival countersignature must use {ALG_SLH_DSA_SHA2_128S}; "
            f"got {slh_dsa_keypair.suite_id!r}"
        )
    suite = get_suite(ALG_SLH_DSA_SHA2_128S)
    payload = signable_payload(statement)

    # If a verification_method is provided, derive the signature's kid
    # from its id fragment so the entry and VM line up by default. The
    # explicit ``kid`` parameter still wins when the caller really wants
    # a specific kid (rare).
    resolved_kid = kid
    if verification_method and kid == "archival":
        vm_id = verification_method.get("id") or ""
        if "#" in vm_id:
            resolved_kid = vm_id.rsplit("#", 1)[-1]

    sig_entry = {
        "alg": suite.alg_id,
        "kid": resolved_kid,
        "sig": _hex_signature(suite, payload, slh_dsa_keypair.private_key),
    }
    return replace(
        statement,
        archival_signature=sig_entry,
        archival_verification_method=(
            dict(verification_method)
            if verification_method
            else statement.archival_verification_method
        ),
    )


def finalize(statement: SuccessionStatement) -> SuccessionStatement:
    """Stamp ``statement_id`` and ``created_at`` after signing is done.

    Both fields are excluded from the signable payload so calling this
    AFTER all signatures are applied does not invalidate them. Calling
    it BEFORE is also fine (id is content-addressed; created_at is
    informational); the convention is "sign, then finalize."
    """
    return replace(
        statement,
        statement_id=compute_statement_id(statement),
        created_at=(
            statement.created_at
            or datetime.now(timezone.utc).isoformat()
        ),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuccessionVerifyResult:
    """Outcome of :func:`verify_succession`.

    ``ok`` is the composite verdict — predecessor + successor (+ archival
    if present) all satisfied their per-side policy.

    Per-side ``PolicyResult`` instances are exposed so callers can log
    or re-check finer-grained outcomes. The ``statement_id_consistent``
    flag flags integrity check on the embedded statement_id.
    """

    ok: bool
    predecessor: PolicyResult
    successor: PolicyResult
    archival: Optional[PolicyResult]
    statement_id_consistent: bool
    reason: str


def _verify_signatures_against(
    payload: bytes,
    signatures: Iterable[Mapping[str, str]],
    verification_methods: Iterable[Mapping[str, Any]],
) -> List[dict]:
    """Drop entries that don't crypto-verify; return survivors.

    Mirrors the same crypto-check logic as
    :func:`identity.hybrid_keypair.verify_hybrid` but isolated here
    for clarity and to avoid a circular import.
    """
    methods_by_kid: dict = {}
    for vm in verification_methods:
        vm_id = vm.get("id") or ""
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        methods_by_kid[kid] = vm

    verified: List[dict] = []
    for entry in signatures:
        if not isinstance(entry, Mapping):
            continue
        alg = entry.get("alg")
        kid = entry.get("kid")
        sig_hex = entry.get("sig")
        if not alg or not kid or not sig_hex:
            continue
        vm = methods_by_kid.get(kid)
        if not vm:
            continue
        multibase = vm.get("publicKeyMultibase")
        if not isinstance(multibase, str):
            continue
        try:
            suite, pub = multibase_to_public_key(multibase)
        except CryptoSuiteError:
            continue
        # alg must match the resolved suite — defends against a kid spoof
        # where an entry claims a different algorithm than the verification
        # method's actual key type.
        if suite.alg_id != alg:
            continue
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            continue
        if suite.verify(payload, sig_bytes, pub):
            verified.append(dict(entry))
    return verified


def verify_succession(
    statement: SuccessionStatement,
    *,
    predecessor_policy: VerifyPolicy = VerifyPolicy.LEGACY_ALLOWED,
    successor_policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    require_archival: bool = False,
) -> SuccessionVerifyResult:
    """Verify a succession statement end-to-end.

    Three independent checks:

    1. **Predecessor**: signatures crypto-verify against
       ``predecessor_verification_methods``, surviving set satisfies
       ``predecessor_policy``. Default ``LEGACY_ALLOWED`` because the
       most common Wave 3 case is a legacy ECDSA-only agent rotating
       to hybrid — they cannot produce a hybrid signature.
    2. **Successor**: same crypto check against successor methods,
       policy ``HYBRID_REQUIRED`` (the new identity IS hybrid by
       construction; if the successor signature is classical-only
       the rotation is malformed and we reject).
    3. **Archival** (if present or ``require_archival=True``): the
       single SLH-DSA-SHA2-128s entry must verify against
       ``archival_verification_method``.

    The composite ``ok`` is True only if all required checks pass.
    A consistency check on ``statement_id`` (must equal
    :func:`compute_statement_id`) runs alongside; a mismatch sets
    ``statement_id_consistent=False`` and forces ``ok=False`` (the
    statement was tampered after signing or never finalized).
    """
    payload = signable_payload(statement)

    # Predecessor side
    pred_verified = _verify_signatures_against(
        payload,
        statement.predecessor_signatures,
        statement.predecessor_verification_methods,
    )
    predecessor_result = evaluate_signatures(pred_verified, predecessor_policy)

    # Successor side
    succ_verified = _verify_signatures_against(
        payload,
        statement.successor_signatures,
        statement.successor_verification_methods,
    )
    successor_result = evaluate_signatures(succ_verified, successor_policy)

    # Archival side
    archival_result: Optional[PolicyResult] = None
    if require_archival or statement.archival_signature:
        if not statement.archival_signature:
            archival_result = PolicyResult(
                ok=False,
                reason="archival countersignature required but not present",
                alg_ids_seen=frozenset(),
            )
        elif not statement.archival_verification_method:
            archival_result = PolicyResult(
                ok=False,
                reason="archival_verification_method missing",
                alg_ids_seen=frozenset(),
            )
        else:
            arch_verified = _verify_signatures_against(
                payload,
                [statement.archival_signature],
                [statement.archival_verification_method],
            )
            archival_result = evaluate_signatures(
                arch_verified, VerifyPolicy.PQ_REQUIRED,
            )
            # Guard: archival sig must specifically be SLH-DSA-128s, not
            # any post-quantum suite. Otherwise a future caller could
            # countersign with ML-DSA-65 and still pass PQ_REQUIRED.
            if archival_result.ok and ALG_SLH_DSA_SHA2_128S not in archival_result.alg_ids_seen:
                archival_result = PolicyResult(
                    ok=False,
                    reason=(
                        f"archival signature must use {ALG_SLH_DSA_SHA2_128S}; "
                        f"got {sorted(archival_result.alg_ids_seen)}"
                    ),
                    alg_ids_seen=archival_result.alg_ids_seen,
                )

    # Statement-id integrity
    expected_id = compute_statement_id(statement)
    id_ok = (not statement.statement_id) or statement.statement_id == expected_id

    composite_ok = predecessor_result.ok and successor_result.ok and id_ok
    if archival_result is not None:
        composite_ok = composite_ok and archival_result.ok

    if composite_ok:
        reason = "succession statement verified"
    else:
        parts = []
        if not predecessor_result.ok:
            parts.append(f"predecessor: {predecessor_result.reason}")
        if not successor_result.ok:
            parts.append(f"successor: {successor_result.reason}")
        if archival_result and not archival_result.ok:
            parts.append(f"archival: {archival_result.reason}")
        if not id_ok:
            parts.append(
                f"statement_id mismatch: stored={statement.statement_id!r} "
                f"computed={expected_id!r}"
            )
        reason = "; ".join(parts) or "unknown failure"

    return SuccessionVerifyResult(
        ok=composite_ok,
        predecessor=predecessor_result,
        successor=successor_result,
        archival=archival_result,
        statement_id_consistent=id_ok,
        reason=reason,
    )


__all__ = [
    "SuccessionStatement",
    "SuccessionVerifyResult",
    "archival_countersign",
    "compute_statement_id",
    "finalize",
    "sign_predecessor",
    "sign_successor",
    "signable_payload",
    "verify_succession",
]
