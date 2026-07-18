"""Portable trust evidence for signed identity packages.

An identity package may carry public keys, but public keys supplied by the
thing being verified are not, by themselves, a trust anchor.  This module
defines the narrow exception that makes offline migration safe:

* a self-certifying ``did:pkh`` / ``did:key`` root may bootstrap itself;
* a ``did:web`` root must be pinned by receiver-owned policy; and
* every later key is authorized by a fully signed succession chain.

The bundle never contains private key material.  Revocation is deliberately
receiver-owned policy rather than an assertion inside the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from kestrel_sovereign.identity.succession import (
    SuccessionStatement,
    compute_statement_id,
    verify_did_binding,
)
from kestrel_sovereign.identity.succession_chain import (
    SuccessionChainError,
    build_chain,
    resolve_active_identity,
    verify_artifact_against_chain,
)
from kestrel_sovereign.security.verify_policy import VerifyPolicy


PORTABLE_TRUST_VERSION = "1.0"
MAX_SUCCESSION_LINKS = 16
MAX_VERIFICATION_METHODS = 8


@dataclass(frozen=True)
class IdentityTrustPolicy:
    """Receiver-owned trust and revocation policy.

    ``trusted_root_verification_methods`` is mandatory for a ``did:web``
    root.  Merely repeating a package's did:web DID in ``trusted_root_did``
    does not pin its keys and is therefore insufficient.
    """

    trusted_root_did: Optional[str] = None
    trusted_root_verification_methods: Tuple[Mapping[str, object], ...] = field(
        default_factory=tuple
    )
    revoked_succession_ids: frozenset[str] = field(default_factory=frozenset)
    require_archival: bool = False
    trusted_archival_multibase: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        trusted_root_did: Optional[str] = None,
        trusted_root_verification_methods: Optional[
            Sequence[Mapping[str, object]]
        ] = None,
        revoked_succession_ids: Optional[Iterable[str]] = None,
        require_archival: bool = False,
        trusted_archival_multibase: Optional[str] = None,
    ) -> "IdentityTrustPolicy":
        return cls(
            trusted_root_did=trusted_root_did,
            trusted_root_verification_methods=tuple(
                dict(vm) for vm in (trusted_root_verification_methods or ())
            ),
            revoked_succession_ids=frozenset(revoked_succession_ids or ()),
            require_archival=require_archival,
            trusted_archival_multibase=trusted_archival_multibase,
        )


def build_identity_trust_bundle(identity) -> dict:
    """Return the public, chain-bound trust bundle for ``identity``."""
    if identity.succession_chain and identity.succession_chain.statements:
        first = identity.succession_chain.statements[0]
        return {
            "version": PORTABLE_TRUST_VERSION,
            "root_did": first.predecessor_did,
            "root_verification_methods": [
                dict(vm) for vm in first.predecessor_verification_methods
            ],
            "successions": [
                statement.to_dict()
                for statement in identity.succession_chain.statements
            ],
        }

    if identity.is_born_hybrid:
        return {
            "version": PORTABLE_TRUST_VERSION,
            "root_did": identity.new_did,
            "root_verification_methods": [
                dict(vm) for vm in (identity.new_verification_methods or ())
            ],
            "successions": [],
        }

    # Legacy did:pkh packages are also made portable.  The DID binds the
    # public key cryptographically, so no receiver-side private custody or
    # pre-existing DID document is needed.
    from kestrel_sovereign.identity.did_web import build_verification_methods
    from kestrel_sovereign.security.crypto_suite import get_suite

    if identity.legacy_did is None or identity.legacy_keypair is None:
        raise ValueError("identity has no portable trust root")
    suite = get_suite(identity.legacy_keypair.suite_id)
    return {
        "version": PORTABLE_TRUST_VERSION,
        "root_did": identity.legacy_did,
        "root_verification_methods": build_verification_methods(
            identity.legacy_did,
            [(suite, identity.legacy_keypair.public_key)],
            kid_prefix="keys",
        ),
        "successions": [],
    }


def _vm_keys(vms: Sequence[Mapping[str, object]]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for index, vm in enumerate(vms):
        if not isinstance(vm, Mapping):
            raise ValueError(f"verification method[{index}] must be an object")
        vm_id = vm.get("id")
        multibase = vm.get("publicKeyMultibase")
        if not isinstance(vm_id, str) or not vm_id:
            raise ValueError(
                f"verification method[{index}] id must be a non-empty string"
            )
        if not isinstance(multibase, str) or not multibase:
            raise ValueError(
                f"verification method[{index}] publicKeyMultibase must be a "
                "non-empty string"
            )
        keys.append((vm_id, multibase))
    return sorted(keys)


def _parse_bundle(package) -> tuple[
    str, list[Mapping[str, object]], tuple[SuccessionStatement, ...]
]:
    bundle = package.identity_trust
    if not isinstance(bundle, Mapping):
        raise ValueError("package has no portable identity trust bundle")
    if bundle.get("version") != PORTABLE_TRUST_VERSION:
        raise ValueError(
            f"unsupported portable trust bundle version {bundle.get('version')!r}"
        )

    root_did = bundle.get("root_did")
    if not isinstance(root_did, str) or not root_did.startswith("did:"):
        raise ValueError("portable trust root_did must be a DID string")
    root_vms = bundle.get("root_verification_methods")
    if not isinstance(root_vms, list) or not root_vms:
        raise ValueError("portable trust root has no verification methods")
    if len(root_vms) > MAX_VERIFICATION_METHODS:
        raise ValueError("portable trust root has too many verification methods")
    if any(not isinstance(vm, Mapping) for vm in root_vms):
        raise ValueError("portable trust root verification methods must be objects")

    raw_successions = bundle.get("successions")
    if not isinstance(raw_successions, list):
        raise ValueError("portable trust successions must be an array")
    if len(raw_successions) > MAX_SUCCESSION_LINKS:
        raise ValueError(
            f"portable trust chain exceeds {MAX_SUCCESSION_LINKS} links"
        )
    statements = []
    for index, raw in enumerate(raw_successions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"portable trust succession[{index}] must be an object")
        statement = SuccessionStatement.from_dict(raw)
        for side, vms in (
            ("predecessor", statement.predecessor_verification_methods),
            ("successor", statement.successor_verification_methods),
        ):
            if not vms or len(vms) > MAX_VERIFICATION_METHODS:
                raise ValueError(
                    f"portable trust succession[{index}] {side} has an "
                    "invalid verification-method count"
                )
        statements.append(statement)
    return root_did, list(root_vms), tuple(statements)


def verify_portable_package(package, policy: Optional[IdentityTrustPolicy] = None):
    """Verify a signed package using only public, chain-bound evidence.

    Returns the usual ``(ok, message)`` verifier contract.  The success
    message names both the root and the succession statement IDs used.
    """
    try:
        return _verify_portable_package(package, policy)
    except Exception as exc:
        # This is a public verifier over attacker-controlled package data.
        # Malformed collections must produce a structured rejection rather
        # than escaping as TypeError/KeyError into direct callers.
        return False, f"Portable trust verification failed closed: {exc}"


def _verify_portable_package(
    package,
    policy: Optional[IdentityTrustPolicy] = None,
):
    try:
        root_did, root_vms, statements = _parse_bundle(package)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"Portable trust bundle rejected: {exc}"

    policy = policy or IdentityTrustPolicy()
    if policy.trusted_root_did is not None and policy.trusted_root_did != root_did:
        return False, (
            f"Portable trust root mismatch: policy pins "
            f"{policy.trusted_root_did!r}, package claims {root_did!r}"
        )
    if package.did != root_did:
        return False, (
            f"Portable trust bundle is for root {root_did!r}, not package DID "
            f"{package.did!r}"
        )

    pinned_vms = list(policy.trusted_root_verification_methods)
    if root_did.startswith("did:web:"):
        if not pinned_vms:
            return False, (
                "Portable did:web root is self-declared and is not trusted. "
                "Receiver policy must pin its root verification methods."
            )
        if _vm_keys(pinned_vms) != _vm_keys(root_vms):
            return False, "Portable did:web root verification methods do not match policy pin"
        root_resolver = lambda did: {
            "id": root_did,
            "verificationMethod": [dict(vm) for vm in pinned_vms],
        } if did == root_did else (_raise_unresolved(did))
        root_ok, root_reason = verify_did_binding(
            root_did, root_vms, did_web_resolver=root_resolver
        )
    else:
        root_ok, root_reason = verify_did_binding(root_did, root_vms)
        if root_ok and pinned_vms and _vm_keys(pinned_vms) != _vm_keys(root_vms):
            return False, "Portable root verification methods do not match policy pin"
    if not root_ok:
        return False, f"Portable trust root binding failed: {root_reason}"

    revoked = policy.revoked_succession_ids
    for index, statement in enumerate(statements):
        canonical_id = compute_statement_id(statement)
        if canonical_id in revoked or statement.statement_id in revoked:
            return False, (
                f"Portable trust succession[{index}] {canonical_id} is revoked"
            )

    try:
        chain = build_chain(statements)
    except SuccessionChainError as exc:
        return False, f"Portable trust chain rejected: {exc}"

    # In this context offline did:web resolution is safe: the trusted root
    # authenticates the first link, and each statement commits both the next
    # DID and its exact VMs with predecessor authorization plus successor
    # acceptance.  Reject ambiguous duplicate DID material.
    docs: dict[str, list[Mapping[str, object]]] = {root_did: root_vms}
    seen_successor_dids: set[str] = {root_did}
    for statement in statements:
        if statement.successor_did in seen_successor_dids:
            return False, (
                f"Portable trust chain repeats DID {statement.successor_did!r}; "
                "cyclic succession is not allowed"
            )
        seen_successor_dids.add(statement.successor_did)
        for did, vms in (
            (statement.predecessor_did, statement.predecessor_verification_methods),
            (statement.successor_did, statement.successor_verification_methods),
        ):
            prior = docs.get(did)
            if prior is not None and _vm_keys(prior) != _vm_keys(vms):
                return False, f"Portable trust chain gives conflicting keys for {did!r}"
            docs[did] = list(vms)

    def offline_resolver(did: str):
        vms = docs.get(did)
        if vms is None:
            return _raise_unresolved(did)
        return {"id": did, "verificationMethod": [dict(vm) for vm in vms]}

    active_vms = resolve_active_identity(
        root_did=root_did,
        root_verification_methods=root_vms,
        chain=chain,
        artifact_timestamp=package.export_timestamp,
    ).verification_methods
    if _vm_keys(package.verification_methods or []) != _vm_keys(active_vms):
        return False, (
            "Package verification methods do not match the active identity "
            "authorized by the portable trust chain"
        )

    artifact_signatures = []
    for signature in package.iter_signatures():
        entry = dict(signature)
        kid = entry.get("kid")
        if isinstance(kid, str) and "#" in kid:
            entry["kid"] = kid.rsplit("#", 1)[-1]
        artifact_signatures.append(entry)

    has_hybrid = any(
        isinstance(sig, Mapping)
        and sig.get("alg") in ("ed25519", "ml-dsa-65")
        for sig in artifact_signatures
    )
    try:
        result = verify_artifact_against_chain(
            root_did=root_did,
            root_verification_methods=root_vms,
            chain=chain,
            artifact_timestamp=package.export_timestamp,
            artifact_payload=package.content_hash.encode("utf-8"),
            artifact_signatures=artifact_signatures,
            policy=(
                VerifyPolicy.HYBRID_REQUIRED
                if has_hybrid
                else VerifyPolicy.LEGACY_ALLOWED
            ),
            require_archival=policy.require_archival,
            trusted_archival_multibase=policy.trusted_archival_multibase,
            did_web_resolver=offline_resolver,
        )
    except Exception as exc:
        return False, f"Portable trust verification failed closed: {exc}"
    if not result.ok:
        return False, f"Portable trust verification failed: {result.reason}"

    evidence = [compute_statement_id(statement) for statement in statements]
    evidence_text = ",".join(evidence) if evidence else "self-certifying root"
    return True, (
        f"Signature valid (portable; trust_root={root_did}; "
        f"chain_evidence={evidence_text})"
    )


def _raise_unresolved(did: str):
    raise ValueError(f"portable trust resolver has no evidence for {did!r}")


__all__ = [
    "IdentityTrustPolicy",
    "MAX_SUCCESSION_LINKS",
    "PORTABLE_TRUST_VERSION",
    "build_identity_trust_bundle",
    "verify_portable_package",
]
