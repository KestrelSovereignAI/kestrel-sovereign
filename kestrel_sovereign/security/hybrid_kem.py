"""
Hybrid KEM combiner — Wave 4 sub-PR 2 of Quantum Hardening (#921, #919).

Composes two independent KEMs (classical X25519 + post-quantum
ML-KEM-768) into one hybrid KEM whose security holds as long as
EITHER underlying KEM is secure.

Construction (concatenation hybrid)
-----------------------------------

We follow the IETF concatenation-hybrid design (draft-ietf-pquip-
hybrid-kems, "concat" combiner) which is also what CNSA 2.0 prescribes:

1. Encapsulator runs ``encap_classical(pk_C)`` and ``encap_pq(pk_PQ)``
   independently to get ``(ct_C, ss_C)`` and ``(ct_PQ, ss_PQ)``.
2. Hybrid ciphertext is the **concatenation** ``ct_C || ct_PQ``.
3. Hybrid shared secret is the output of an **HKDF-SHA256 extract+
   expand** over ``ss_C || ss_PQ``, salted with the *transcript*
   ``ct_C || ct_PQ || pk_C || pk_PQ``. Salting with the transcript
   pins the derived key to the exact ciphertexts and recipient
   public keys, preventing key-reuse / strong-multi-recipient
   degradation attacks. Output length is configurable; the default
   is 32 bytes (the AES-256-GCM key size).
4. Decapsulator splits the ciphertext, runs both decapsulations
   independently, recomputes the same KDF input, and gets back
   the same hybrid shared secret.

Forward safety
--------------

If X25519 falls to a Shor-equipped adversary later, ML-KEM-768's
shared-secret entropy keeps the HKDF output secret. If a
cryptanalytic break of Module-LWE is discovered, X25519's ECDH
entropy still protects the HKDF output. The KDF is a one-way
extraction; recovering the AES-256 key would require breaking
HKDF-SHA256 itself, which currently has no faster attack than
2^128 (Grover) brute force.

What this module does NOT do
----------------------------

- It does not encrypt the wrapped material. The caller takes the
  derived 32-byte secret and feeds it to AES-256-GCM (Wave 0C's
  ``AEADCipher``) or any other AEAD. The AEAD's ciphertext is the
  payload an attacker would also have to break.
- It does not handle key rotation. A new keypair → a new ciphertext;
  the wire format does not embed a key id. Higher layers (Wave 4
  sub-PR 3 / capsule format) are responsible for binding hybrid
  ciphertexts to the recipient's identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from kestrel_sovereign.security.kem_suite import (
    ALG_ML_KEM_768,
    ALG_X25519,
    KEMKeypair,
    KEMSuiteError,
    get_kem_suite,
)


# Hybrid combiner tag prefix for the HKDF info parameter. The actual
# info bound to each derivation also includes the selected suite ids
# (see _build_hkdf_info) so two different algorithm pairs (e.g.
# X25519+ML-KEM-768 vs X25519+ML-KEM-1024) cannot accidentally collide
# in their derived keys.
_HKDF_INFO_PREFIX = b"kestrel-hybrid-kem-v1: HKDF-SHA256(ss_C||ss_PQ); algs="

DEFAULT_DERIVED_SECRET_BYTES = 32  # AES-256 key size


# ---------------------------------------------------------------------------
# Hybrid keypair container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridKEMKeypair:
    """A composite KEM keypair: classical + post-quantum together.

    Both halves are independent ``KEMKeypair`` instances. A hybrid
    "public key" is the pair ``(classical.public_key, pq.public_key)``;
    a hybrid "private key" is the pair of private keys. We don't
    fold them into single byte strings here — the consumer wraps both
    sides in DID keyAgreement entries (or bundles them in the capsule
    format Wave 4 sub-PR 3 will define) and keeps the structural
    separation visible.
    """

    classical: KEMKeypair
    pq: KEMKeypair


def generate_hybrid_kem_keypair(
    *,
    classical_alg: str = ALG_X25519,
    pq_alg: str = ALG_ML_KEM_768,
) -> HybridKEMKeypair:
    """Generate both halves of a hybrid KEM keypair.

    Refuses pairs that are both classical or both post-quantum: the
    whole point of the construction is one of each. Algorithm-pair
    validation is the same defensive idea as
    ``identity.hybrid_keypair.generate_hybrid_keypair`` (Wave 2).
    """
    classical = get_kem_suite(classical_alg)
    pq = get_kem_suite(pq_alg)
    if classical.is_post_quantum:
        raise KEMSuiteError(
            f"hybrid classical_alg must be a classical KEM; "
            f"{classical_alg!r} is post-quantum"
        )
    if not pq.is_post_quantum:
        raise KEMSuiteError(
            f"hybrid pq_alg must be a post-quantum KEM; "
            f"{pq_alg!r} is classical"
        )
    return HybridKEMKeypair(
        classical=classical.generate_keypair(),
        pq=pq.generate_keypair(),
    )


# ---------------------------------------------------------------------------
# Hybrid encapsulation / decapsulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridKEMCiphertext:
    """Bundle of the two underlying ciphertexts.

    Carrying them as separate fields (rather than a single
    concatenated blob) makes the wire format self-documenting and
    lets the decapsulator length-check each half against its suite's
    declared ``CIPHERTEXT_SIZE`` rather than relying on a fixed split
    point. The KDF still binds them via the transcript (see
    :func:`_derive_secret`).
    """

    classical_ct: bytes
    pq_ct: bytes


def _build_hkdf_info(classical_alg: str, pq_alg: str) -> bytes:
    """Domain-separation tag that includes the actual selected suite ids.

    Codex P2: ``_HKDF_INFO_PREFIX`` alone hardcoded the default pair,
    so a future suite combination (e.g. X25519 + ML-KEM-1024) could
    derive the same key as the default pair if everything else matched.
    Binding the algorithm ids into the info ensures every distinct
    pair lives in its own domain.
    """
    return _HKDF_INFO_PREFIX + classical_alg.encode("utf-8") + b"+" + pq_alg.encode("utf-8")


def _derive_secret(
    classical_ss: bytes,
    pq_ss: bytes,
    classical_ct: bytes,
    pq_ct: bytes,
    classical_pub: bytes,
    pq_pub: bytes,
    classical_alg: str,
    pq_alg: str,
    out_len: int,
) -> bytes:
    """HKDF-SHA256 extract+expand over the concatenation of both shared
    secrets, salted with the full encapsulation transcript.

    Why the transcript salt
    -----------------------

    HKDF's ``salt`` parameter is not optional for hybrid KEMs: per
    draft-ietf-pquip-hybrid-kems, including the ciphertexts and
    public keys in the salt input prevents an attacker from
    pre-computing useful HKDF outputs across recipients (a multi-
    recipient malleability concern that doesn't bite a vanilla KEM
    but DOES bite a careless concat hybrid).

    Why the info pin
    ----------------

    The info string (``_HKDF_INFO``) carries the combiner-spec
    identifier so a future ``v2`` combiner cannot accidentally
    produce the same key as ``v1`` for the same inputs.
    """
    ikm = bytes(classical_ss) + bytes(pq_ss)
    salt = bytes(classical_ct) + bytes(pq_ct) + bytes(classical_pub) + bytes(pq_pub)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=out_len,
        salt=salt,
        info=_build_hkdf_info(classical_alg, pq_alg),
        backend=default_backend(),
    )
    return hkdf.derive(ikm)


def encapsulate_hybrid(
    classical_public_key: Any,
    pq_public_key: Any,
    *,
    classical_alg: str = ALG_X25519,
    pq_alg: str = ALG_ML_KEM_768,
    out_len: int = DEFAULT_DERIVED_SECRET_BYTES,
) -> Tuple[HybridKEMCiphertext, bytes]:
    """Encapsulate a fresh hybrid shared secret to a recipient.

    Args:
        classical_public_key: recipient's classical KEM public key
            (X25519 by default — pass an ``X25519PublicKey`` or
            equivalent).
        pq_public_key: recipient's post-quantum public key (ML-KEM-768
            raw bytes by default).
        classical_alg / pq_alg: override the default suite pair.
        out_len: derived-secret length in bytes. Default 32 = AES-256
            key. Larger values fan out via HKDF expand for multiple
            keys (e.g. AES key + integrity key + IV) without
            increasing entropy cost.

    Returns:
        ``(HybridKEMCiphertext, derived_secret_bytes)``. The caller
        sends the ciphertext to the recipient and uses the secret as
        an AEAD key.
    """
    if out_len <= 0:
        raise KEMSuiteError(f"out_len must be positive; got {out_len}")

    classical = get_kem_suite(classical_alg)
    pq = get_kem_suite(pq_alg)

    if classical.is_post_quantum:
        raise KEMSuiteError(
            f"hybrid classical_alg must be a classical KEM; got {classical_alg!r}"
        )
    if not pq.is_post_quantum:
        raise KEMSuiteError(
            f"hybrid pq_alg must be a post-quantum KEM; got {pq_alg!r}"
        )

    classical_ct, classical_ss = classical.encapsulate(classical_public_key)
    pq_ct, pq_ss = pq.encapsulate(pq_public_key)

    # Serialize public keys for the transcript salt. Both halves go
    # through their suite's serializer so we get the canonical wire
    # bytes (the same bytes a verifier would feed back through
    # deserialize_public_key).
    classical_pub_bytes = classical.serialize_public_key(classical_public_key)
    pq_pub_bytes = pq.serialize_public_key(pq_public_key)

    secret = _derive_secret(
        classical_ss, pq_ss,
        classical_ct, pq_ct,
        classical_pub_bytes, pq_pub_bytes,
        classical_alg, pq_alg,
        out_len,
    )
    return HybridKEMCiphertext(classical_ct=classical_ct, pq_ct=pq_ct), secret


def decapsulate_hybrid(
    ciphertext: HybridKEMCiphertext,
    classical_keypair: KEMKeypair,
    pq_keypair: KEMKeypair,
    *,
    out_len: int = DEFAULT_DERIVED_SECRET_BYTES,
) -> bytes:
    """Recover the hybrid shared secret encapsulated to this recipient.

    Both halves of the keypair are needed; the recipient must hold
    both private keys. Returns the same derived secret the
    encapsulator computed iff both ciphertexts are unmodified AND
    both private keys correspond to the public keys the encapsulator
    used.

    Failure semantics: the underlying suites' wrong-key behavior
    propagates through. ML-KEM-768 implicitly returns a different
    shared secret on wrong-key (FIPS 203), and so the HKDF output
    differs and the AEAD downstream fails authentication. X25519
    likewise returns a different ECDH product. There is no explicit
    "wrong key" exception at this layer — the AEAD is the
    authentication boundary.
    """
    if out_len <= 0:
        raise KEMSuiteError(f"out_len must be positive; got {out_len}")

    classical = get_kem_suite(classical_keypair.suite_id)
    pq = get_kem_suite(pq_keypair.suite_id)

    if classical.is_post_quantum:
        raise KEMSuiteError(
            f"classical_keypair.suite_id={classical_keypair.suite_id!r} is "
            f"post-quantum; expected a classical KEM"
        )
    if not pq.is_post_quantum:
        raise KEMSuiteError(
            f"pq_keypair.suite_id={pq_keypair.suite_id!r} is classical; "
            f"expected a post-quantum KEM"
        )

    classical_ss = classical.decapsulate(ciphertext.classical_ct, classical_keypair.private_key)
    pq_ss = pq.decapsulate(ciphertext.pq_ct, pq_keypair.private_key)

    classical_pub_bytes = classical.serialize_public_key(classical_keypair.public_key)
    pq_pub_bytes = pq.serialize_public_key(pq_keypair.public_key)

    return _derive_secret(
        classical_ss, pq_ss,
        ciphertext.classical_ct, ciphertext.pq_ct,
        classical_pub_bytes, pq_pub_bytes,
        classical_keypair.suite_id, pq_keypair.suite_id,
        out_len,
    )


__all__ = [
    "DEFAULT_DERIVED_SECRET_BYTES",
    "HybridKEMCiphertext",
    "HybridKEMKeypair",
    "decapsulate_hybrid",
    "encapsulate_hybrid",
    "generate_hybrid_kem_keypair",
]
