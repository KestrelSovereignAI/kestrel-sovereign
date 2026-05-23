"""
Data-access grants — owner-signed consent for cross-agent imports
(kestrel-sovereign #1273, prerequisite for kestrel-feature-healthcare
Phase D / Frinz #163).

A **data-access grant** is the cryptographic gate that lets an owner
authorize a specific source agent's package to be imported into a
specific host agent. Without it, sovereign import is implicitly
platform-authorized — anyone running a Kestrel host can ingest any
package as long as its signature verifies. That defeats the patient-
as-root-of-trust model the healthcare epic requires.

The grant attests:

- ``owner_did``  — the data subject (e.g. the patient)
- ``source_did`` — the exporting agent whose package is authorized
- ``host_did``   — the receiving agent (where the records will land)
- ``issued_at``  — when the grant was issued (UTC ISO 8601)
- ``expires_at`` — optional expiry; empty means no expiry
- ``purpose``    — free-text scope/purpose description
- ``owner_verification_methods`` — the owner's keys at signing time
- ``owner_signatures``           — the owner's signatures over the
  canonical payload

Why the owner signs, not the source
-----------------------------------

Source signatures already exist on the package itself (the source
agent attests "this is my export"). A grant signed by the *source*
would prove nothing the package signature doesn't already prove. The
missing primitive is owner authorization — *the data subject* saying
"I authorize this source's records to be imported into this host."
That requires the owner's keys, not the source's.

Why ``host_did`` binds to the receiving agent's own DID
-------------------------------------------------------

Kestrel does not maintain a platform-level DID distinct from the
agent. The canonical identity at import time is the receiving agent's
own ``did`` (per the Agent Identity Contract). Binding the grant to
that DID gives the owner the *strongest* control: a grant authorizing
import into agent ``did:pkh:…:abc`` cannot be replayed against a
different agent ``did:pkh:…:xyz`` running on the same Kestrel
deployment. A platform-level host DID would actually be weaker — it
would let any co-tenant agent reuse the same grant.

Distinct rejection reasons
--------------------------

:class:`ConsentVerification` exposes five named boolean fields, each
corresponding to one acceptance-criterion check. Callers can surface
the precise failure mode in audit logs:

  * ``package_signed_by_source``   — package signature verifies under
                                     the source DID's bound VMs
  * ``grant_signed_by_owner``      — grant signature verifies under
                                     the owner DID's bound VMs
  * ``grant_names_source``         — grant.source_did equals the
                                     package's DID
  * ``grant_targets_host``         — grant.host_did equals the
                                     receiving agent's DID
  * ``grant_not_expired_or_revoked`` — grant.revoked is False and the
                                     optional expires_at has not passed

Host policy is OUT of consent scope. A valid grant is necessary;
host-side allowlists are an additional optional filter the caller
layers on top of a ``ok=True`` result, never a substitute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.identity.succession import verify_did_binding
from kestrel_sovereign.security.crypto_suite import (
    CryptoSuite,
    CryptoSuiteError,
    Keypair,
    get_suite,
)
from kestrel_sovereign.security.multikey import multibase_to_public_key

# A ``did:web:`` resolver: ``Callable[[str did], Mapping[str, Any]]``
# returning the DID document (typically the same shape published at
# ``https://<domain>/.well-known/did.json``). Callers MUST pass one
# explicitly when accepting did:web owners — refusing-by-default
# prevents an attacker from getting a "free pass" by claiming a
# did:web that no resolver knows about.
DidWebResolver = "Callable[[str], Mapping[str, Any]]"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataAccessGrant:
    """An immutable owner-signed authorization to import data.

    Every signed field is committed in the canonical signable payload
    (see :func:`signable_payload`). ``grant_id`` is content-addressed
    by SHA-256 over that payload, so any change to a signed field
    changes the id; ``created_at`` is informational and excluded from
    the payload so timestamp drift between the signer and the archiver
    doesn't invalidate the signature — the binding timestamps are
    ``issued_at`` and the optional ``expires_at``, both of which ARE
    in the signed payload.

    There is NO ``revoked`` field. Revocation is a runtime state of
    the issuing context, not part of the issued credential, and an
    in-grant flag would be unsigned / spoofable (codex P2 #1273 R2).
    Revocation is supplied to :func:`verify_import_consent` via the
    ``revoked_grant_ids`` set, sourced from a trusted registry.

    ``grant_id`` is carried on the dataclass for human-readable
    diagnostics only. The verifier ALWAYS recomputes the canonical id
    from the signable payload and exposes it on
    :class:`ConsentVerification` — host policies and audit logs must
    use ``ConsentVerification.canonical_grant_id`` rather than this
    field, which a caller could spoof at the serialization boundary.
    """

    owner_did: str
    source_did: str
    host_did: str
    issued_at: str
    expires_at: str = ""
    purpose: str = ""
    owner_verification_methods: List[dict] = field(default_factory=list)
    owner_signatures: List[dict] = field(default_factory=list)
    grant_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """Plain-dict representation for JSON archival.

        All list and dict fields are deep-copied; the result is safe
        to mutate without affecting the underlying frozen dataclass.
        """
        return {
            "owner_did": self.owner_did,
            "source_did": self.source_did,
            "host_did": self.host_did,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "purpose": self.purpose,
            "owner_verification_methods": [
                dict(m) for m in self.owner_verification_methods
            ],
            "owner_signatures": [dict(s) for s in self.owner_signatures],
            "grant_id": self.grant_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataAccessGrant":
        return cls(
            owner_did=data["owner_did"],
            source_did=data["source_did"],
            host_did=data["host_did"],
            issued_at=data["issued_at"],
            expires_at=data.get("expires_at", ""),
            purpose=data.get("purpose", ""),
            owner_verification_methods=list(
                data.get("owner_verification_methods") or []
            ),
            owner_signatures=list(data.get("owner_signatures") or []),
            grant_id=data.get("grant_id", ""),
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Canonical signable payload
# ---------------------------------------------------------------------------
#
# Every owner signature is computed over EXACTLY this byte sequence.
# Excludes:
#   - owner_signatures — circular.
#   - grant_id         — derived from this payload, also circular.
#   - created_at       — informational; binding times are issued_at /
#                        expires_at, both of which ARE included.
#   - revoked          — revocation is a runtime state, not part of
#                        the issued credential. A revoked grant whose
#                        revoked=True still signs-verifies; revocation
#                        is enforced at verify time, not by re-signing.

_GRANT_SIGNED_FIELDS = (
    "owner_did",
    "source_did",
    "host_did",
    "issued_at",
    "expires_at",
    "purpose",
    "owner_verification_methods",
)


def signable_payload(grant: DataAccessGrant) -> bytes:
    """Canonical UTF-8 bytes for signing/verification.

    Sorted-key compact JSON ensures byte-stability across signers and
    verifiers regardless of runtime dict-iteration order, so a
    verifier re-deriving the bytes from a ``to_dict()`` round-trip
    obtains the exact same bytes the signer used.
    """
    payload = {
        f: getattr(grant, f)
        for f in _GRANT_SIGNED_FIELDS
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def compute_grant_id(grant: DataAccessGrant) -> str:
    """SHA-256 hex of :func:`signable_payload`.

    Stable content-addressed id: any change to a signed field changes
    the id. Audit logs reference grants by this id so a swap of one
    grant for another can't slip past a consumer that recorded the id.
    """
    return hashlib.sha256(signable_payload(grant)).hexdigest()


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

def _hex_signature(suite: CryptoSuite, data: bytes, private_key: Any) -> str:
    return suite.sign(data, private_key).hex()


def sign_owner(
    grant: DataAccessGrant,
    owner_keypairs: Iterable[Tuple[Keypair, str]],
) -> DataAccessGrant:
    """Apply owner signatures over :func:`signable_payload`.

    Each ``(keypair, kid)`` pair signs the canonical payload and is
    appended to ``owner_signatures``. ``kid`` MUST match the fragment
    of one of ``owner_verification_methods[].id`` so verification can
    route the signature to the correct VM. A legacy owner with a
    secp256k1-only key passes exactly one pair; a hybrid owner passes
    two (Ed25519 + ML-DSA-65).
    """
    payload = signable_payload(grant)
    sigs = list(grant.owner_signatures)
    for kp, kid in owner_keypairs:
        suite = get_suite(kp.suite_id)
        sigs.append({
            "alg": suite.alg_id,
            "kid": kid,
            "sig": _hex_signature(suite, payload, kp.private_key),
        })
    return replace(grant, owner_signatures=sigs)


def finalize(grant: DataAccessGrant) -> DataAccessGrant:
    """Stamp ``grant_id`` and ``created_at`` after signing is done.

    Both fields are excluded from the signable payload so calling this
    AFTER signing does not invalidate the signatures. Convention is
    "sign, then finalize."
    """
    return replace(
        grant,
        grant_id=compute_grant_id(grant),
        created_at=(
            grant.created_at or datetime.now(timezone.utc).isoformat()
        ),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsentVerification:
    """Outcome of :func:`verify_import_consent`.

    ``ok`` is the composite verdict — every named check below passed.
    Each per-check boolean is independently surfaceable so callers can
    record precise failure reasons in audit logs (the
    ``agent_import_log.reject_reason`` field uses these names).

    ``canonical_grant_id`` is the content-addressed id the verifier
    recomputed from the signable payload — host policies and audit
    logs MUST reference this rather than ``grant.grant_id`` (which a
    caller could spoof at the serialization boundary).

    ``reason`` is a human-readable joined explanation of failing
    checks, suitable for log lines and error messages.
    """

    ok: bool
    package_signed_by_source: bool
    grant_signed_by_owner: bool
    grant_names_source: bool
    grant_targets_host: bool
    grant_not_expired_or_revoked: bool
    canonical_grant_id: str
    reason: str


# Distinct rejection reason codes. Surfaced directly so audit logs and
# downstream consumers can branch on a stable identifier.
REJECT_PACKAGE_SIGNATURE = "package_signature_invalid"
REJECT_GRANT_SIGNATURE = "grant_signature_invalid"
REJECT_GRANT_NAMES_DIFFERENT_SOURCE = "grant_names_different_source"
REJECT_GRANT_TARGETS_DIFFERENT_HOST = "grant_targets_different_host"
REJECT_GRANT_EXPIRED_OR_REVOKED = "grant_expired_or_revoked"
REJECT_HOST_POLICY = "host_policy_rejected"


def _parse_iso_utc(s: str) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    candidate = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    if dt.utcoffset() != timedelta(0):
        # Non-UTC offset — same UTC-only contract as succession.
        return None
    return dt


def _verify_owner_signatures(
    grant: DataAccessGrant,
    *,
    did_web_resolver: Optional[Any] = None,
) -> Tuple[bool, str]:
    """At least one ``owner_signature`` must crypto-verify against the
    owner's embedded verification methods, AND the owner DID must
    bind cryptographically to those VMs (so an attacker can't embed
    their own VMs alongside the owner's DID).

    ``did_web_resolver`` is forwarded to :func:`verify_did_binding`.
    Without it, ``did:web:`` owners are refused fail-closed (the
    documented refuse-by-default contract on the binding helper).
    """
    if not grant.owner_signatures:
        return False, "grant has no owner_signatures"
    if not grant.owner_verification_methods:
        return False, "grant has no owner_verification_methods"

    bound_ok, bound_reason = verify_did_binding(
        grant.owner_did,
        grant.owner_verification_methods,
        did_web_resolver=did_web_resolver,
    )
    if not bound_ok:
        return False, f"owner DID not bound to its VMs: {bound_reason}"

    methods_by_kid: dict = {}
    for vm in grant.owner_verification_methods:
        if not isinstance(vm, Mapping):
            continue
        vm_id = vm.get("id") or ""
        kid = vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id
        if kid:
            methods_by_kid[kid] = vm

    payload = signable_payload(grant)
    any_verified = False
    for entry in grant.owner_signatures:
        # Tolerate malformed serialized input: a non-mapping entry
        # (e.g. a stray string from bad deserialization) is just an
        # invalid signature — skip it rather than crashing the
        # caller with AttributeError on .get(). codex P2 #1273 R3.
        if not isinstance(entry, Mapping):
            continue
        alg = entry.get("alg")
        kid = entry.get("kid")
        sig_hex = entry.get("sig")
        if not (alg and kid and sig_hex):
            continue
        vm = methods_by_kid.get(kid)
        if vm is None:
            continue
        multibase = vm.get("publicKeyMultibase")
        if not isinstance(multibase, str):
            continue
        try:
            suite, public_key = multibase_to_public_key(multibase)
        except CryptoSuiteError:
            continue
        if suite.alg_id != alg:
            # Signature was signed by alg A but the VM carries alg B —
            # an attacker substituting a different VM under the same
            # kid would land here. Reject this attempt.
            continue
        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            continue
        try:
            if suite.verify(payload, sig_bytes, public_key):
                any_verified = True
                break
        except Exception:
            continue

    if not any_verified:
        return False, "no owner_signature verified against bound VMs"
    return True, ""


async def _verify_package_signed_by_source(package: Any) -> Tuple[bool, str]:
    """Verify the inbound package was signed by ``package.did``.

    Delegates to :func:`identity.signing.verify_package_signature` so
    consent verification routes through the same hybrid/legacy logic
    every other Kestrel signature check uses.
    """
    # Imported lazily so this module stays free of a hard runtime
    # dep on signing.py at import time — verification is the only
    # call site that needs it.
    from kestrel_sovereign.identity.signing import verify_package_signature

    try:
        ok, reason = verify_package_signature(package)
    except Exception as e:
        return False, f"package signature check raised: {e}"
    return ok, reason


async def verify_import_consent(
    package: Any,
    grant: DataAccessGrant,
    *,
    host_did: str,
    revoked_grant_ids: Optional[Iterable[str]] = None,
    did_web_resolver: Optional[Any] = None,
    now: Optional[datetime] = None,
) -> ConsentVerification:
    """Verify a :class:`DataAccessGrant` authorizes importing *package*
    into the agent identified by *host_did*.

    Runs five named checks (the acceptance-criteria contract) and
    composes a :class:`ConsentVerification` result. The function does
    NOT short-circuit on the first failure — every check runs so audit
    consumers see the full surface, which is useful when a single
    misconfiguration produces multiple symptoms (e.g. a tampered
    grant trips both signature AND host-binding checks).

    Args:
        package: The inbound package whose ``did``, ``signature`` /
            ``signatures``, and ``verification_methods`` fields are
            inspected. Any object whose attribute shape matches
            :class:`AgentIdentityPackage` works.
        grant: The owner-signed :class:`DataAccessGrant`.
        host_did: The receiving agent's own DID. The grant's
            ``host_did`` field MUST equal this; otherwise an attacker
            could replay a grant minted for one agent against another
            on the same deployment.
        revoked_grant_ids: Optional iterable of canonical grant ids
            (as returned by :func:`compute_grant_id`) that are
            currently revoked. If the recomputed canonical id of
            ``grant`` is in this set, the grant is rejected with
            ``grant_expired_or_revoked``. Revocation lives OUTSIDE
            the grant payload by design — an in-grant flag would be
            unsigned and trivially spoofable in serialized flows
            (codex P2 #1273 R2). Sourced from a trusted registry.
        did_web_resolver: Optional resolver passed through to
            :func:`verify_did_binding` for ``did:web:`` owners. The
            binding helper refuses-by-default when an owner_did is
            ``did:web:`` and no resolver is provided — without this
            parameter, hybrid (Ed25519 + ML-DSA-65) owners on
            ``did:web:`` would have their grants rejected as
            ``grant_signature_invalid`` even when correctly signed.
            ``did:pkh:`` / ``did:key:`` owners need no resolver.
        now: Optional clock override (UTC datetime). Defaults to
            ``datetime.now(timezone.utc)``. Tests pass a fixed value.
    """
    # Always recompute the content-addressed id from the signable
    # payload — never trust ``grant.grant_id`` (unsigned, spoofable
    # at the serialization boundary; codex P2 #1273 R2).
    canonical_grant_id = compute_grant_id(grant)

    # Check 1 — package signed by its declared source DID.
    pkg_ok, pkg_reason = await _verify_package_signed_by_source(package)

    # Check 2 — grant signed by its declared owner DID.
    grant_ok, grant_reason = _verify_owner_signatures(
        grant, did_web_resolver=did_web_resolver,
    )

    # Check 3 — grant's source_did matches the package's DID.
    pkg_did = getattr(package, "did", None)
    names_source = bool(pkg_did) and grant.source_did == pkg_did

    # Check 4 — grant's host_did matches the receiving agent's DID.
    targets_host = bool(host_did) and grant.host_did == host_did

    # Check 5 — grant not revoked (per external registry) and not
    # expired. Revocation is checked against the recomputed canonical
    # id; expiry is checked against ``grant.expires_at`` which IS in
    # the signed payload.
    revocation_ok = True
    revocation_detail = ""
    if revoked_grant_ids is not None:
        revoked_set = set(revoked_grant_ids)
        if canonical_grant_id in revoked_set:
            revocation_ok = False
            revocation_detail = (
                f"canonical grant_id {canonical_grant_id[:16]}… is in "
                f"the revocation set"
            )
    expiry_ok = True
    expiry_detail = ""
    if grant.expires_at:
        exp = _parse_iso_utc(grant.expires_at)
        if exp is None:
            expiry_ok = False
            expiry_detail = (
                f"expires_at={grant.expires_at!r} is malformed"
            )
        else:
            current = now if now is not None else datetime.now(timezone.utc)
            if exp <= current:
                expiry_ok = False
                expiry_detail = (
                    f"expires_at={grant.expires_at!r} is in the past"
                )
    not_expired_or_revoked = revocation_ok and expiry_ok

    reasons: List[str] = []
    if not pkg_ok:
        reasons.append(f"{REJECT_PACKAGE_SIGNATURE}: {pkg_reason}")
    if not grant_ok:
        reasons.append(f"{REJECT_GRANT_SIGNATURE}: {grant_reason}")
    if not names_source:
        reasons.append(
            f"{REJECT_GRANT_NAMES_DIFFERENT_SOURCE}: grant names "
            f"source_did={grant.source_did!r} but package is "
            f"signed by did={pkg_did!r}"
        )
    if not targets_host:
        reasons.append(
            f"{REJECT_GRANT_TARGETS_DIFFERENT_HOST}: grant targets "
            f"host_did={grant.host_did!r} but receiving agent is "
            f"did={host_did!r}"
        )
    if not not_expired_or_revoked:
        detail = revocation_detail or expiry_detail or "expired or revoked"
        reasons.append(f"{REJECT_GRANT_EXPIRED_OR_REVOKED}: {detail}")

    ok = (
        pkg_ok
        and grant_ok
        and names_source
        and targets_host
        and not_expired_or_revoked
    )
    return ConsentVerification(
        ok=ok,
        package_signed_by_source=pkg_ok,
        grant_signed_by_owner=grant_ok,
        grant_names_source=names_source,
        grant_targets_host=targets_host,
        grant_not_expired_or_revoked=not_expired_or_revoked,
        canonical_grant_id=canonical_grant_id,
        reason="; ".join(reasons) if reasons else "consent verified",
    )


__all__ = [
    "DataAccessGrant",
    "ConsentVerification",
    "REJECT_PACKAGE_SIGNATURE",
    "REJECT_GRANT_SIGNATURE",
    "REJECT_GRANT_NAMES_DIFFERENT_SOURCE",
    "REJECT_GRANT_TARGETS_DIFFERENT_HOST",
    "REJECT_GRANT_EXPIRED_OR_REVOKED",
    "REJECT_HOST_POLICY",
    "signable_payload",
    "compute_grant_id",
    "sign_owner",
    "finalize",
    "verify_import_consent",
]
