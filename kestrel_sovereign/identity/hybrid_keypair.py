"""
HybridKeypair — composite classical + post-quantum identity (Wave 2 sub-PR 4).

Quantum Hardening epic #921, sub-issue #917. Implements the hybrid-signing
shape called out by CNSA 2.0: every identity assertion carries BOTH a
classical signature (Ed25519, decades-aged cryptanalysis) AND a
post-quantum signature (ML-DSA-65, NIST FIPS 204 Cat-3). A verifier under
``HYBRID_REQUIRED`` policy accepts the artifact only if both halves
verify; under ``LEGACY_ALLOWED`` either half alone suffices through the
migration window. See :mod:`kestrel_sovereign.security.verify_policy`.

Why two signatures, not one composite signature
-----------------------------------------------

CNSA 2.0 + IETF (draft-ietf-pquip-hybrid-signature-spec) explicitly
recommend the *concatenation* hybrid form (two independent signatures
side-by-side) over composite forms (e.g. binding both into one
serialized blob). Concatenation has three properties we want:

1. **Forward safety**: if either half is broken cryptographically, the
   other still bounds an attacker's success — an attacker has to forge
   *both* simultaneously to produce a verifying hybrid signature.
2. **Algorithm agility**: the verifier can independently reject one
   half (e.g. classical-only post-cutoff via ``post_cutoff_classical_
   allowed=False``) without the other half being entangled in the same
   serialization.
3. **Deployability**: each half is a standard signature consumed by
   off-the-shelf verifiers (Ed25519 verifiers don't need to know about
   ML-DSA-65 at all). Wave 5's release-signing path can carry a hybrid
   signature whose classical half is verifiable by tools that haven't
   been patched yet.

The signatures are written into the v2 ``signatures`` array as TWO
independent entries with distinct ``alg`` and ``kid`` fields, exactly
what ``identity_package_v2.compute_content_hash`` already excludes from
its hash.

Default algorithm pair
----------------------

Ed25519 + ML-DSA-65. Both halves chosen for hybrid-pair properties:

- Classical: Ed25519 (deterministic, RFC 8032, in widespread Web/UCAN
  deployment, library-stable).
- PQ: ML-DSA-65 (NIST Cat-3, FIPS-204-pinned, prebuilt-wheel
  pqcrypto-backed — no compile-on-deploy).

Either half is overrideable via ``classical_alg`` / ``pq_alg`` for
algorithm migration tests and the future SLH-DSA hash-based path
(Wave 3+).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.security.crypto_suite import (
    ALG_ED25519,
    ALG_ML_DSA_65,
    CryptoSuite,
    CryptoSuiteError,
    Keypair,
    get_suite,
)
from kestrel_sovereign.security.verify_policy import (
    PolicyResult,
    VerifyPolicy,
    evaluate_signatures,
)


DEFAULT_CLASSICAL_ALG = ALG_ED25519
DEFAULT_PQ_ALG = ALG_ML_DSA_65


@dataclass(frozen=True)
class HybridKeypair:
    """A classical + post-quantum keypair pair.

    ``classical`` and ``pq`` are independent ``Keypair`` instances;
    each owns its own ``suite_id``. The hybrid keypair is the *bundle*
    — it has no separate suite of its own. Verifiers iterate the v2
    ``signatures`` array and pair each entry with the matching
    verification method by ``kid``.
    """

    classical: Keypair
    pq: Keypair

    def public_keys(self) -> List[Tuple[CryptoSuite, Any]]:
        """``[(classical_suite, classical_pub), (pq_suite, pq_pub)]``.

        Used by ``did_web.build_did_document`` to emit two Multikey
        verification methods. Order is stable: classical first, then
        post-quantum — matches the kid scheme below and the
        ``signatures`` array order produced by :func:`sign_hybrid`.
        """
        return [
            (get_suite(self.classical.suite_id), self.classical.public_key),
            (get_suite(self.pq.suite_id), self.pq.public_key),
        ]


def generate_hybrid_keypair(
    *,
    classical_alg: str = DEFAULT_CLASSICAL_ALG,
    pq_alg: str = DEFAULT_PQ_ALG,
) -> HybridKeypair:
    """Generate a fresh hybrid keypair.

    Both halves are generated from independent system entropy via the
    suite registry. Default pair is Ed25519 + ML-DSA-65; pass alternate
    suite ids to use Secp256k1 + ML-DSA-65, Ed25519 + SLH-DSA (Wave 3+),
    or any registered (classical, post-quantum) pair.

    Raises ``CryptoSuiteError`` if either suite is unregistered or if
    the pair isn't actually classical+PQ (e.g. two PQ suites — that
    isn't a hybrid signature, it's just two PQ signatures and we
    refuse rather than silently mis-classify).
    """
    classical_suite = get_suite(classical_alg)
    pq_suite = get_suite(pq_alg)
    if classical_suite.is_post_quantum:
        raise CryptoSuiteError(
            f"hybrid classical_alg must be a classical suite; "
            f"{classical_alg!r} is post-quantum"
        )
    if not pq_suite.is_post_quantum:
        raise CryptoSuiteError(
            f"hybrid pq_alg must be a post-quantum suite; "
            f"{pq_alg!r} is classical"
        )
    return HybridKeypair(
        classical=classical_suite.generate_keypair(),
        pq=pq_suite.generate_keypair(),
    )


def sign_hybrid(
    data: bytes,
    keypair: HybridKeypair,
    *,
    classical_kid: str = "key-1",
    pq_kid: str = "key-2",
) -> List[dict]:
    """Sign ``data`` with both halves; return v2 ``signatures`` entries.

    The output drops directly into ``AgentIdentityPackage.signatures``::

        [
            {"alg": "ed25519",   "kid": "key-1", "sig": "<hex>"},
            {"alg": "ml-dsa-65", "kid": "key-2", "sig": "<hex>"},
        ]

    Default kids match the ``did_web.build_verification_methods``
    1-indexed default (``#key-1`` / ``#key-2``) so the signature kids
    line up with the DID document's verification-method ids without
    extra plumbing. Override the kids for stable-rotation schemes
    (e.g. ``#ed25519`` / ``#ml-dsa-65``).
    """
    classical_suite = get_suite(keypair.classical.suite_id)
    pq_suite = get_suite(keypair.pq.suite_id)

    classical_sig = classical_suite.sign(data, keypair.classical.private_key)
    pq_sig = pq_suite.sign(data, keypair.pq.private_key)

    return [
        {"alg": classical_suite.alg_id, "kid": classical_kid, "sig": classical_sig.hex()},
        {"alg": pq_suite.alg_id, "kid": pq_kid, "sig": pq_sig.hex()},
    ]


def verify_hybrid(
    data: bytes,
    signatures: Iterable[Mapping[str, str]],
    verification_methods: Iterable[Mapping[str, Any]],
    *,
    policy: VerifyPolicy = VerifyPolicy.HYBRID_REQUIRED,
    post_cutoff_classical_allowed: bool = True,
) -> PolicyResult:
    """Verify a v2 ``signatures`` array against ``verification_methods``.

    Two-step check:

    1. **Crypto verification.** For each ``signatures`` entry, find the
       matching verification method by ``kid`` (the entry's ``kid``
       must match the fragment of a verification method's ``id``).
       Resolve the method's ``publicKeyMultibase`` to a public key and
       verify the signature against ``data`` using the entry's ``alg``.
       Any failed crypto verification drops that entry from the set
       passed to the policy evaluator — the policy never sees signatures
       that don't actually verify.

    2. **Policy evaluation.** :func:`evaluate_signatures` runs against
       the surviving entries with ``policy`` and
       ``post_cutoff_classical_allowed``. A HYBRID_REQUIRED check that
       passes here means at least one classical and at least one PQ
       signature both cryptographically verified.

    Returns a :class:`PolicyResult` so callers don't have to re-run the
    policy evaluation themselves.
    """
    # Index verification methods by kid (last fragment after '#')
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
            from kestrel_sovereign.security.multikey import multibase_to_public_key
            suite, pub = multibase_to_public_key(multibase)
        except CryptoSuiteError:
            continue

        # Cross-check: signature alg must match the method's algorithm.
        # Otherwise a caller could sign with key-1 (Ed25519) but tag the
        # entry as ml-dsa-65 and sneak past a PQ-required check.
        if suite.alg_id != alg:
            continue

        try:
            sig_bytes = bytes.fromhex(sig_hex)
        except ValueError:
            continue

        if suite.verify(data, sig_bytes, pub):
            verified.append(dict(entry))

    return evaluate_signatures(
        verified,
        policy,
        post_cutoff_classical_allowed=post_cutoff_classical_allowed,
    )


__all__ = [
    "DEFAULT_CLASSICAL_ALG",
    "DEFAULT_PQ_ALG",
    "HybridKeypair",
    "generate_hybrid_keypair",
    "sign_hybrid",
    "verify_hybrid",
]
